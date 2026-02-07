# app.py
from __future__ import annotations

import os
import io
import re
import time
import uuid
import json
import sqlite3
import traceback
import importlib
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

# Optional but recommended for MS OAuth:
import msal


# =========================
# CONFIG
# =========================
APP_TITLE = "Phase 1 Workforce Planning"
DB_PATH = os.getenv("APP_DB_PATH", "app.db")
DATA_DIR = os.getenv("APP_DATA_DIR", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
LOG_DIR = os.path.join(DATA_DIR, "logs")
DEMO_AUTH = os.getenv("DEMO_AUTH", "1").strip() == "1"

# Azure AD (Microsoft 365 SSO) configs:
TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("AZURE_REDIRECT_URI", "")  # must match Azure App Registration redirect URI

OWNER_EMAIL = (os.getenv("OWNER_EMAIL", "") or "").strip().lower()
ALLOWED_DOMAIN = (os.getenv("ALLOWED_DOMAIN", "") or "").strip().lower()  # optional: "@company.com"

# Scopes for basic identity
SCOPES = ["openid", "profile", "email"]

# Model module/file you already have (must expose run_from_excel)
# Accepts values like: planning_6_2_2, planning_6_2-2.py, /abs/path/planning_6_2-2.py
MODEL_MODULE_NAME = os.getenv("MODEL_MODULE", "planning_6_2-2.py")

# =========================
# VALIDATION RULES
# =========================
REQUIRED_SHEETS = ["buckets", "demand", "ta_capacity"]
REQUIRED_BUCKET_COLS = ["zone", "family", "hc0"]
REQUIRED_DEMAND_COLS = ["week", "zone", "family", "pmc_demand", "group_demand"]
REQUIRED_TA_COLS = ["week", "hiring_capacity"]

# Optional but if sheet exists and not empty then must have cols:
TRANSFER_ALLOWED_COLS = ["from_zone", "from_family", "to_zone", "to_family", "allowed"]


# =========================
# UTIL
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs():
    for d in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def norm_col(c: str) -> str:
    return str(c).strip().lower()


def safe_lower(x: Any) -> str:
    return str(x).strip().lower()


def short_id() -> str:
    return uuid.uuid4().hex[:12]


def file_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s.strip()))
def render_logo(
    path: str,
    *,
    sidebar: bool = False,
    width: Optional[int] = None,
    remove_baked_bg: bool = True,
) -> None:

    if not path or not os.path.exists(path):
        return

    image_fn = st.sidebar.image if sidebar else st.image

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(path).convert("RGBA")
        arr = np.array(img).astype(np.uint8)

        # If already has transparency, don't try to remove background
        if remove_baked_bg:
            alpha = arr[..., 3]
            has_transparency = np.any(alpha < 255)

            if not has_transparency:
                r = arr[..., 0].astype(np.int16)
                g = arr[..., 1].astype(np.int16)
                b = arr[..., 2].astype(np.int16)

                mean = (r + g + b) / 3.0
                near_grey = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (np.abs(r - b) < 12)
                bright = mean > 200  # only remove bright greys/whites
                bg = near_grey & bright

                arr[..., 3] = np.where(bg, 0, arr[..., 3])

        out = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)

        if width:
            image_fn(buf, width=width)
        else:
            image_fn(buf, use_container_width=True)

    except Exception:
        if width:
            image_fn(path, width=width)
        else:
            image_fn(path, use_container_width=True)

# =========================
# DB
# =========================
def db_conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def db_init():
    con = db_conn()
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS allowlist (
            email TEXT PRIMARY KEY,
            added_by TEXT,
            added_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            user_email TEXT,
            created_at TEXT,
            status TEXT,
            input_path TEXT,
            output_path TEXT,
            error_message TEXT
        )
    """)

    con.commit()

    # Ensure owner exists in allowlist (not required, but nice)
    if OWNER_EMAIL and is_email(OWNER_EMAIL):
        cur.execute("INSERT OR IGNORE INTO allowlist(email, added_by, added_at) VALUES(?,?,?)",
                    (OWNER_EMAIL, OWNER_EMAIL, now_str()))
        con.commit()

    con.close()


def db_allowlist_get_all() -> List[Tuple[str, str, str]]:
    con = db_conn()
    rows = con.execute("SELECT email, added_by, added_at FROM allowlist ORDER BY added_at DESC").fetchall()
    con.close()
    return rows


def db_allowlist_add(email: str, added_by: str):
    con = db_conn()
    con.execute("INSERT OR REPLACE INTO allowlist(email, added_by, added_at) VALUES(?,?,?)",
                (email, added_by, now_str()))
    con.commit()
    con.close()


def db_allowlist_remove(email: str):
    con = db_conn()
    con.execute("DELETE FROM allowlist WHERE email=?", (email,))
    con.commit()
    con.close()


def db_is_allowed(email: str) -> bool:
    email = safe_lower(email)
    if not email:
        return False
    if OWNER_EMAIL and email == OWNER_EMAIL:
        return True
    if ALLOWED_DOMAIN and email.endswith(ALLOWED_DOMAIN):
        # Domain allowed (optional)
        return True
    con = db_conn()
    row = con.execute("SELECT 1 FROM allowlist WHERE email=?", (email,)).fetchone()
    con.close()
    return row is not None


def db_run_create(run_id: str, user_email: str, input_path: str, output_path: str):
    con = db_conn()
    con.execute("""
        INSERT INTO runs(run_id, user_email, created_at, status, input_path, output_path, error_message)
        VALUES(?,?,?,?,?,?,?)
    """, (run_id, user_email, now_str(), "RUNNING", input_path, output_path, None))
    con.commit()
    con.close()


def db_run_update(run_id: str, status: str, error_message: Optional[str] = None):
    con = db_conn()
    con.execute("UPDATE runs SET status=?, error_message=? WHERE run_id=?",
                (status, error_message, run_id))
    con.commit()
    con.close()


def db_runs_list(limit: int = 50) -> List[Tuple]:
    con = db_conn()
    rows = con.execute("""
        SELECT run_id, user_email, created_at, status, input_path, output_path, error_message
        FROM runs ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return rows


def db_run_get(run_id: str) -> Optional[Tuple]:
    con = db_conn()
    row = con.execute("""
        SELECT run_id, user_email, created_at, status, input_path, output_path, error_message
        FROM runs WHERE run_id=?
    """, (run_id,)).fetchone()
    con.close()
    return row


# =========================
# AUTH (Microsoft 365 SSO)
# =========================
def auth_is_configured() -> bool:
    return all([TENANT_ID, CLIENT_ID, CLIENT_SECRET, REDIRECT_URI])


def msal_app() -> msal.ConfidentialClientApplication:
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    return msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=authority,
        client_credential=CLIENT_SECRET,
    )


def build_auth_url(state: str) -> str:
    app = msal_app()
    return app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        state=state,
        prompt="select_account",
        response_type="code",
    )


def exchange_code_for_token(code: str) -> Dict[str, Any]:
    app = msal_app()
    result = app.acquire_token_by_authorization_code(
        code=code,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    return result


def extract_email_from_idtoken(id_token_claims: Dict[str, Any]) -> str:
    # Azure can put email in different fields
    for k in ["preferred_username", "email", "upn", "unique_name"]:
        v = id_token_claims.get(k)
        if v and isinstance(v, str) and "@" in v:
            return v.strip().lower()
    return ""


def logout():
    # Clear session & query params
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    try:
        st.query_params.clear()
    except Exception:
        pass

def ensure_login():
    # KHÔNG render title "Authentication" nữa

    # --- DEMO MODE (no real Microsoft auth) ---
    if DEMO_AUTH:
        if st.session_state.get("user_email"):
            st.sidebar.success(f"(DEMO) Signed in:\n\n{st.session_state['user_email']}")
            if st.sidebar.button("Logout"):
                logout()
                st.rerun()
            return

        st.sidebar.info("DEMO mode")
        if st.sidebar.button("Sign in with Microsoft"):
            st.session_state["user_email"] = "demo.user@demo.com"
            st.session_state["id_claims"] = {"demo": True}
            st.rerun()
        return

    # --- REAL MODE (Microsoft 365) ---
    if not auth_is_configured():
        st.error(
            "Chưa cấu hình Microsoft 365 SSO.\n\n"
            "Bạn cần set ENV:\n"
            "- AZURE_TENANT_ID\n- AZURE_CLIENT_ID\n- AZURE_CLIENT_SECRET\n- AZURE_REDIRECT_URI\n- OWNER_EMAIL\n"
        )
        st.stop()

    if st.session_state.get("user_email"):
        st.sidebar.success(f"Signed in:\n\n{st.session_state['user_email']}")
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()
        return

    qp = st.query_params
    code = qp.get("code", None)

    if code:
        with st.spinner("Signing in with Microsoft 365..."):
            token_result = exchange_code_for_token(code)
        if "error" in token_result:
            st.error(f"Login failed: {token_result.get('error_description', token_result.get('error'))}")
            st.stop()

        id_claims = token_result.get("id_token_claims") or {}
        email = extract_email_from_idtoken(id_claims)

        if not email:
            st.error("Không đọc được email từ token Microsoft.")
            st.stop()

        st.session_state["user_email"] = email
        st.session_state["id_claims"] = id_claims

        try:
            st.query_params.clear()
        except Exception:
            pass

        st.rerun()

    state_val = st.session_state.get("oauth_state") or short_id()
    st.session_state["oauth_state"] = state_val
    auth_url = build_auth_url(state_val)

    st.sidebar.info("Bạn cần login Microsoft 365 để sử dụng.")
    st.sidebar.link_button("Sign in with Microsoft", auth_url)




def require_authorized_user():
    email = st.session_state.get("user_email", "")
    if not email:
        st.stop()
    return



def is_owner() -> bool:
    email = st.session_state.get("user_email", "")
    return bool(OWNER_EMAIL) and safe_lower(email) == safe_lower(OWNER_EMAIL)


# =========================
# INPUT VALIDATOR
# =========================
@dataclass
class ValidationError:
    sheet: str
    issue: str
    details: str


def validate_excel(file_path: str) -> Tuple[bool, List[ValidationError], Dict[str, pd.DataFrame]]:
    """
    Returns:
      ok, errors, previews (dict sheet->df head)
    """
    errors: List[ValidationError] = []
    previews: Dict[str, pd.DataFrame] = {}

    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        errors.append(ValidationError(sheet="(file)", issue="Cannot open Excel", details=str(e)))
        return False, errors, previews

    # Check required sheets
    sheets_lower = [s.lower() for s in xl.sheet_names]
    for s in REQUIRED_SHEETS:
        if s not in sheets_lower:
            errors.append(ValidationError(sheet=s, issue="Missing required sheet", details=f"Missing sheet '{s}'"))

    if errors:
        return False, errors, previews

    # Helper to load sheet by name case-insensitive
    def read_sheet(name: str) -> pd.DataFrame:
        real = xl.sheet_names[sheets_lower.index(name)]
        df = pd.read_excel(xl, real).dropna(how="all")
        df.columns = [norm_col(c) for c in df.columns]
        return df

    # buckets
    bdf = read_sheet("buckets")
    previews["buckets"] = bdf.head(10)
    for c in REQUIRED_BUCKET_COLS:
        if c not in bdf.columns:
            errors.append(ValidationError(sheet="buckets", issue="Missing required column", details=c))

    # demand
    ddf = read_sheet("demand")
    previews["demand"] = ddf.head(10)
    for c in REQUIRED_DEMAND_COLS:
        if c not in ddf.columns:
            errors.append(ValidationError(sheet="demand", issue="Missing required column", details=c))

    # ta_capacity
    tdf = read_sheet("ta_capacity")
    previews["ta_capacity"] = tdf.head(10)
    for c in REQUIRED_TA_COLS:
        if c not in tdf.columns:
            errors.append(ValidationError(sheet="ta_capacity", issue="Missing required column", details=c))

    # Optional: transfer_allowed
    if "transfer_allowed" in sheets_lower:
        trdf = read_sheet("transfer_allowed")
        previews["transfer_allowed"] = trdf.head(10)
        # If empty => ok (means allow all pairs in your model logic)
        if not trdf.empty:
            for c in TRANSFER_ALLOWED_COLS:
                if c not in trdf.columns:
                    errors.append(ValidationError(sheet="transfer_allowed", issue="Missing column", details=c))

    ok = len(errors) == 0
    return ok, errors, previews


# =========================
# MODEL RUNNER
# =========================
def _load_model_module(model_ref: str):
    """
    Load model from module name or python file path.

    Supports:
      - "planning_6_2_2"
      - "planning_6_2-2.py"
      - "/abs/path/planning_6_2-2.py"
    """
    ref = (model_ref or "").strip()
    if not ref:
        raise RuntimeError("MODEL_MODULE rỗng. Hãy cấu hình tên module hoặc đường dẫn file .py.")

    if ref.endswith(".py") or os.path.sep in ref:
        abs_path = ref if os.path.isabs(ref) else os.path.join(os.getcwd(), ref)
        if not os.path.exists(abs_path):
            raise RuntimeError(
                f"Không tìm thấy file model: {abs_path}. "
                f"Hãy đặt file model đúng đường dẫn hoặc cập nhật MODEL_MODULE."
            )

        mod_name = f"planning_model_{short_id()}"
        spec = importlib.util.spec_from_file_location(mod_name, abs_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Không load được module spec từ file: {abs_path}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    try:
        return importlib.import_module(ref)
    except Exception as e:
        raise RuntimeError(
            f"Không import được module '{ref}'.\n"
            f"- Hãy đảm bảo module nằm trong PYTHONPATH\n"
            f"- Hoặc set MODEL_MODULE thành đường dẫn file .py\n\n"
            f"Import error: {e}"
        )


def run_model(input_path: str, output_path: str) -> None:
    """
    Calls model.run_from_excel(input_path, output_path)
    """
    mod = _load_model_module(MODEL_MODULE_NAME)

    fn = getattr(mod, "run_from_excel", None)
    if fn is None:
        raise RuntimeError(
            f"Model '{MODEL_MODULE_NAME}' không có hàm run_from_excel(input_path, output_path)."
        )

    fn(input_path, output_path)


def read_output_frames(output_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read only the 'data' sheets we need. New output uses:
      - KPI
      - Bucket Weekly (Long)
      - Transfers Long
      - Solve Info
    (GF Dashboard / Transfer Matrix are formatted sheets, not ideal for pandas)
    """
    xl = pd.ExcelFile(output_path)
    sheets = xl.sheet_names

    # prefer new names, but keep backward compatibility
    wanted = []
    for s in ["KPI", "Bucket Weekly (Long)", "Transfers Long", "Solve Info"]:
        if s in sheets:
            wanted.append(s)

    # fallback old names (if you still open old outputs)
    for s in ["Bucket_Weekly", "Transfers_Long", "Solve_Info"]:
        if s in sheets and s not in wanted:
            wanted.append(s)

    if not wanted:
        # fallback: read all (not recommended but safe)
        return pd.read_excel(output_path, sheet_name=None)

    return pd.read_excel(xl, sheet_name=wanted)

def _get_frame(frames: Dict[str, pd.DataFrame], candidates: List[str]) -> pd.DataFrame:
    for k in candidates:
        if k in frames and frames[k] is not None and not frames[k].empty:
            return frames[k]
    return pd.DataFrame()


def _normalize_bucket_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # map new output columns -> internal canonical columns
    colmap = {
        "Week": "week",
        "Zone": "zone",
        "Family": "family",
        "GF": "gf",
        "PMC Demand": "pmc",
        "UR": "ur",
        "Production Demand": "pd",
        "Actual": "actual",
        "Turnover Rate": "to_rate",
        "Absent": "absent_rate",
        "HC net": "hc_net",                   # NOTE: in Bucket Weekly = AFTER transfer (as per your model)
        "Hiring": "hire",
        "Slack HC": "slack_hc",
        "In": "transfer_in",
        "Out": "transfer_out",
        "Gap": "gap",
        # optional columns (you added)
        "HC net (no transfer)": "hc_net_no_transfer",
        "Gap (no transfer)": "gap_no_transfer",
    }
    df2 = df.copy()
    for old, new in colmap.items():
        if old in df2.columns and new not in df2.columns:
            df2 = df2.rename(columns={old: new})

    # make sure key cols exist
    for c in ["week", "zone", "family"]:
        if c not in df2.columns:
            # try old naming style
            if c in df.columns:
                continue
            return pd.DataFrame()

    # ensure numeric where needed
    for c in ["pd", "hc_net", "hire", "slack_hc", "actual", "gap", "transfer_in", "transfer_out"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)

    df2["week"] = pd.to_numeric(df2["week"], errors="coerce").fillna(0).astype(int)
    df2["zone"] = df2["zone"].astype(str)
    df2["family"] = df2["family"].astype(str)
    if "gf" in df2.columns:
        df2["gf"] = df2["gf"].astype(str)
    else:
        df2["gf"] = df2["zone"] + " | " + df2["family"]

    return df2


def _normalize_transfers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    colmap = {
        "Week": "week",
        "From GF": "from_gf",
        "To GF": "to_gf",
        "Transfer HC": "transfer_hc",
    }
    df2 = df.copy()
    for old, new in colmap.items():
        if old in df2.columns and new not in df2.columns:
            df2 = df2.rename(columns={old: new})

    if "week" in df2.columns:
        df2["week"] = pd.to_numeric(df2["week"], errors="coerce").fillna(0).astype(int)
    if "transfer_hc" in df2.columns:
        df2["transfer_hc"] = pd.to_numeric(df2["transfer_hc"], errors="coerce").fillna(0.0)

    return df2


# =========================
# UI PAGES
# =========================
def page_home_upload():
    st.header("Upload & Validate")

    st.markdown(
        "- Upload file Excel đúng template\n"
        "- App sẽ check **sheet bắt buộc** và báo lỗi\n"
        "- Nếu OK → Bấm **Run Model**"
    )

    up = st.file_uploader("Upload input .xlsx", type=["xlsx"])
    if not up:
        st.info("Chưa upload file.")
        return

    # Save uploaded to disk
    run_id = st.session_state.get("draft_run_id") or short_id()
    st.session_state["draft_run_id"] = run_id
    input_path = os.path.join(UPLOAD_DIR, f"{run_id}_input.xlsx")
    with open(input_path, "wb") as f:
        f.write(up.getbuffer())

    ok, errors, previews = validate_excel(input_path)

    st.subheader("Validation Result")
    if ok:
        st.success("File hợp lệ ✅")
    else:
        st.error("File không hợp lệ ❌")
        df_err = pd.DataFrame([{"sheet": e.sheet, "issue": e.issue, "details": e.details} for e in errors])
        st.dataframe(df_err, use_container_width=True)

    with st.expander("Preview sheets (top 10 rows)", expanded=False):
        for name, df in previews.items():
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True)

    # Run button only if ok
    st.divider()
    st.subheader("▶️ Run")
    if not ok:
        st.warning("Sửa lỗi validation trước khi Run.")
        return

    # Default output name avoids Excel lock by using timestamp
    out_name = f"{run_id}_output_{file_timestamp()}.xlsx"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    if st.button("Run Model", type="primary"):
        user_email = st.session_state["user_email"]
        db_run_create(run_id=run_id, user_email=user_email, input_path=input_path, output_path=output_path)

        log_path = os.path.join(LOG_DIR, f"{run_id}.log")
        try:
            with st.spinner("Running model..."):
                # Write a simple log
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(f"[{now_str()}] START run_id={run_id} user={user_email}\n")
                    lf.write(f"[{now_str()}] input={input_path}\n")
                    lf.write(f"[{now_str()}] output={output_path}\n")

                run_model(input_path=input_path, output_path=output_path)

                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"[{now_str()}] DONE\n")

            db_run_update(run_id, status="DONE", error_message=None)
            st.success("Run thành công ✅")
            st.session_state["last_run_id"] = run_id

            # Download immediately
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download output.xlsx",
                    data=f.read(),
                    file_name=os.path.basename(output_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.info("Sang tab Dashboard để xem biểu đồ.")
        except Exception as e:
            tb = traceback.format_exc()
            db_run_update(run_id, status="FAILED", error_message=str(e))
            st.error("Run thất bại ❌")
            st.code(tb)

            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_str()}] FAILED: {e}\n")
                lf.write(tb + "\n")


def page_dashboard():
    st.header("📊 Dashboard")

    runs = db_runs_list(limit=50)
    if not runs:
        st.info("Chưa có run nào.")
        return

    run_ids = [r[0] for r in runs]
    default = st.session_state.get("last_run_id")
    idx = run_ids.index(default) if default in run_ids else 0

    choice = st.selectbox("Chọn Run", run_ids, index=idx)
    row = db_run_get(choice)
    if not row:
        st.error("Run không tồn tại.")
        return

    run_id, user_email, created_at, status, input_path, output_path, error_message = row
    st.caption(f"Run: {run_id} | User: {user_email} | Time: {created_at} | Status: {status}")

    if status != "DONE":
        st.warning("Run chưa DONE. Nếu FAILED thì xem History để đọc error.")
        return

    if not os.path.exists(output_path):
        st.error("Không tìm thấy output file trên disk.")
        return

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download output.xlsx",
            data=f.read(),
            file_name=os.path.basename(output_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    frames = read_output_frames(output_path)

    kpi = _get_frame(frames, ["KPI"])
    bw_raw = _get_frame(frames, ["Bucket Weekly (Long)", "Bucket_Weekly"])
    tr_raw = _get_frame(frames, ["Transfers Long", "Transfers_Long"])
    solve_info = _get_frame(frames, ["Solve Info", "Solve_Info"])

    bw = _normalize_bucket_weekly(bw_raw)
    transfers = _normalize_transfers(tr_raw)

    # ===== KPI =====
    st.subheader("KPI")
    if not kpi.empty:
        k = kpi.iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Production Demand", f"{float(k.get('total_production_demand', 0)):,.0f}")
        c2.metric("Total Slack HC", f"{float(k.get('total_slack_hc', 0)):,.2f}")
        c3.metric("Total Hiring", f"{float(k.get('total_hiring', 0)):,.0f}")
        c4.metric("Total Transfer", f"{float(k.get('total_transfer', 0)):,.0f}")
    else:
        st.info("Không có sheet KPI.")

    if not solve_info.empty:
        with st.expander("Solve Info", expanded=False):
            st.dataframe(solve_info, use_container_width=True)

    st.divider()

    if bw.empty:
        st.warning("Không có sheet Bucket Weekly (Long) / Bucket_Weekly.")
        return

    # ===== Filters =====
    st.subheader("Filter")
    weeks = sorted(bw["week"].dropna().unique().tolist())
    zones = sorted(bw["zone"].dropna().unique().tolist())
    fams = sorted(bw["family"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    w_sel = col1.multiselect("Week", weeks, default=weeks[: min(3, len(weeks))] if weeks else [])
    z_sel = col2.multiselect("Zone", zones, default=zones)
    f_sel = col3.multiselect("Family", fams, default=fams)

    df = bw.copy()
    if w_sel:
        df = df[df["week"].isin(w_sel)]
    if z_sel:
        df = df[df["zone"].isin(z_sel)]
    if f_sel:
        df = df[df["family"].isin(f_sel)]

    st.subheader("Bucket Weekly (filtered)")
    st.dataframe(df, use_container_width=True)

    # ===== Charts =====
    st.subheader("Charts (overall)")

    # Aggregate weekly
    agg_cols = {}
    for c in ["pd", "hc_net", "hire", "slack_hc", "gap"]:
        if c in bw.columns:
            agg_cols[c] = "sum"

    agg = bw.groupby("week", as_index=False).agg(agg_cols)

    def _bold_line_chart(df: pd.DataFrame, metrics: list[str], title: str, height: int = 320):
        if df.empty or not all(m in df.columns for m in metrics):
            st.info(f"Thiếu cột để vẽ: {metrics}")
            return

        d = df[["week"] + metrics].melt("week", var_name="metric", value_name="value")

        chart = (
            alt.Chart(d)
            .mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=4)
            .encode(
                x=alt.X("week:O", title="Week", axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
                y=alt.Y("value:Q", title="People", axis=alt.Axis(labelFontSize=12, titleFontSize=14, grid=True)),
                color=alt.Color("metric:N", legend=alt.Legend(title=None, labelFontSize=12)),
                tooltip=[
                    alt.Tooltip("week:O", title="Week"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value", format=",.1f"),
                ],
            )
            .properties(height=height, title=alt.TitleParams(title, fontSize=18, fontWeight="bold"))
            .configure_view(strokeWidth=0)
            .configure_axis(gridOpacity=0.25)
        )

        st.altair_chart(chart, use_container_width=True)

    def _bold_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, height: int = 280):
        if df.empty or x not in df.columns or y not in df.columns:
            st.info(f"Thiếu cột để vẽ: {x}, {y}")
            return

        chart = (
            alt.Chart(df)
            .mark_bar(size=32, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X(f"{x}:O", title="Week", axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
                y=alt.Y(f"{y}:Q", title=title, axis=alt.Axis(labelFontSize=12, titleFontSize=14, grid=True)),
                tooltip=[
                    alt.Tooltip(f"{x}:O", title="Week"),
                    alt.Tooltip(f"{y}:Q", title="Value", format=",.1f"),
                ],
            )
            .properties(height=height, title=alt.TitleParams(title, fontSize=18, fontWeight="bold"))
            .configure_view(strokeWidth=0)
            .configure_axis(gridOpacity=0.25)
        )

        st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        _bold_line_chart(agg, ["hc_net", "pd"], "Production Demand vs HC net (after transfer)", height=320)

    with c2:
        # hiring/slack may be missing in some outputs; chart will warn gracefully
        _bold_line_chart(agg, [c for c in ["hire", "slack_hc"] if c in agg.columns], "Hiring / Slack HC", height=320)

    st.divider()

    # Transfer volume: make it a BAR to look punchier
    if not transfers.empty and {"week", "transfer_hc"}.issubset(transfers.columns):
        tr_agg = transfers.groupby("week", as_index=False)["transfer_hc"].sum()
        _bold_bar_chart(tr_agg, "week", "transfer_hc", "Transfer volume by week", height=260)

    # ✅ NEW CHART (not duplicate): GAP bar (surplus vs shortage)
    if "gap" in agg.columns:
        gap_df = agg[["week", "gap"]].copy()
        gap_df["sign"] = np.where(gap_df["gap"] >= 0, "Surplus", "Shortage")

        # 1) bar chart (KHÔNG configure ở đây)
        bar = (
            alt.Chart(gap_df)
            .mark_bar(size=32, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("week:O", title="Week", axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
                y=alt.Y(
                    "gap:Q",
                    title="Gap (HC net - PD)",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14, grid=True),
                ),
                color=alt.Color(
                    "sign:N",
                    legend=alt.Legend(title=None, labelFontSize=12),
                    scale=alt.Scale(domain=["Surplus", "Shortage"], range=["#2E7D32", "#C62828"]),
                ),
                tooltip=[
                    alt.Tooltip("week:O", title="Week"),
                    alt.Tooltip("gap:Q", title="Gap", format=",.1f"),
                    alt.Tooltip("sign:N", title=""),
                ],
            )
        )

        # 2) zero line (KHÔNG configure ở đây)
        zero = (
            alt.Chart(pd.DataFrame({"gap": [0]}))
            .mark_rule(strokeWidth=2)
            .encode(y="gap:Q")
        )

        # 3) layer xong MỚI configure ở LayerChart
        gap_chart = (
            alt.layer(bar, zero)
            .properties(
                height=280,
                title=alt.TitleParams("Total Gap by week (HC net - PD)", fontSize=18, fontWeight="bold"),
            )
            .configure_view(strokeWidth=0)
            .configure_axis(gridOpacity=0.25)
        )

        st.altair_chart(gap_chart, use_container_width=True)

    # ===== Transfers table =====
    st.subheader("Transfers")
    if transfers.empty:
        st.info("Không có transfer (Transfers Long rỗng).")
    else:
        st.dataframe(transfers, use_container_width=True)

def page_history():
    st.header("🕒 History")

    runs = db_runs_list(limit=100)
    if not runs:
        st.info("Chưa có run nào.")
        return

    df = pd.DataFrame(runs, columns=["run_id", "user_email", "created_at", "status", "input_path", "output_path", "error_message"])
    st.dataframe(df, use_container_width=True)

    st.subheader("Chi tiết 1 run")
    run_id = st.text_input("Nhập run_id để xem chi tiết", value=st.session_state.get("last_run_id", ""))
    if not run_id:
        return

    row = db_run_get(run_id.strip())
    if not row:
        st.warning("Không tìm thấy run_id.")
        return

    run_id, user_email, created_at, status, input_path, output_path, error_message = row
    st.write({"run_id": run_id, "user_email": user_email, "created_at": created_at, "status": status})

    if status == "FAILED" and error_message:
        st.error("Error message")
        st.code(error_message)

    if output_path and os.path.exists(output_path) and status == "DONE":
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download output.xlsx",
                data=f.read(),
                file_name=os.path.basename(output_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Output chưa sẵn sàng hoặc không tồn tại.")

    # Show log tail if exists
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    if os.path.exists(log_path):
        with st.expander("View log", expanded=False):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                txt = lf.read()[-4000:]  # tail
            st.code(txt)


def page_admin():
    st.header("🛡️ Admin (Owner only)")

    if not is_owner():
        st.error("Chỉ OWNER mới vào được trang này.")
        st.stop()

    st.success(f"OWNER: {OWNER_EMAIL}")

    st.subheader("Allowlist")
    rows = db_allowlist_get_all()
    df = pd.DataFrame(rows, columns=["email", "added_by", "added_at"])
    st.dataframe(df, use_container_width=True)

    st.subheader("Add user")
    new_email = st.text_input("Email cần cấp quyền", placeholder="user@company.com").strip().lower()
    if st.button("Add to allowlist"):
        if not is_email(new_email):
            st.error("Email không hợp lệ.")
        else:
            db_allowlist_add(new_email, st.session_state["user_email"])
            st.success(f"Đã add: {new_email}")
            st.rerun()

    st.subheader("Remove user")
    rm_email = st.text_input("Email cần xoá quyền", placeholder="user@company.com").strip().lower()
    if st.button("Remove from allowlist"):
        if not is_email(rm_email):
            st.error("Email không hợp lệ.")
        else:
            if rm_email == OWNER_EMAIL:
                st.warning("Không nên xoá OWNER khỏi allowlist.")
            db_allowlist_remove(rm_email)
            st.success(f"Đã remove: {rm_email}")
            st.rerun()

    st.subheader("App config quick check")
    st.code(json.dumps({
        "MODEL_MODULE": MODEL_MODULE_NAME,
        "TENANT_ID_set": bool(TENANT_ID),
        "CLIENT_ID_set": bool(CLIENT_ID),
        "CLIENT_SECRET_set": bool(CLIENT_SECRET),
        "REDIRECT_URI": REDIRECT_URI,
        "OWNER_EMAIL": OWNER_EMAIL,
        "ALLOWED_DOMAIN": ALLOWED_DOMAIN,
        "DB_PATH": DB_PATH,
        "DATA_DIR": DATA_DIR,
    }, indent=2))


def page_help():
    st.header("❓ Help")

    st.markdown("""
**Luồng chuẩn:**
1) Login Microsoft 365  
2) Upload Excel đúng template  
3) Validate OK → Run  
4) Download output + xem dashboard

**Các lỗi hay gặp:**
- `PermissionError` khi ghi output.xlsx trên Windows: thường do bạn đang mở file output.xlsx trong Excel.  
  → Giải pháp: đóng Excel hoặc để app tạo output có timestamp (code đã làm rồi).

- Thiếu cột `group_demand` trong sheet `demand`  
  → Đảm bảo header đúng chính tả, không có dấu cách thừa.

- Không có transfer:
  - Nếu sheet `transfer_allowed` có dữ liệu nhưng **allowed=0** hoặc zone/family không match buckets → sẽ không có edge
  - Nếu `transfer_allowed` để trống hoàn toàn → model (theo logic bạn viết) sẽ allow all (trừ khi allow_within_zone_transfer=False)

**Template cứng (bắt buộc):**
- buckets: zone, family, hc0 (optional: turnover_rate, absent_rate, op_per_group, group_count)
- demand: week, zone, family, pmc_demand, group_demand, (optional ur_rate hoặc ur_multiplier)
- ta_capacity: week, hiring_capacity
""")

    st.subheader("Validation rules (app check)")
    st.code(json.dumps({
        "required_sheets": REQUIRED_SHEETS,
        "buckets_required_cols": REQUIRED_BUCKET_COLS,
        "demand_required_cols": REQUIRED_DEMAND_COLS,
        "ta_capacity_required_cols": REQUIRED_TA_COLS,
        "transfer_allowed_cols_if_not_empty": TRANSFER_ALLOWED_COLS,
    }, indent=2))


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown(
        """
        <style>
        /* page padding */
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }

        /* --- Sidebar layout: menu top, auth bottom --- */
        section[data-testid="stSidebar"] .block-container{
          display:flex;
          flex-direction:column;
          height:100vh;
          padding-top: 1rem;
          overflow-y:auto;
        }
        /* Push the LAST element in sidebar to bottom */
        section[data-testid="stSidebar"] .block-container > div:last-child{
          margin-top:auto;
        }

        /* App title: avoid wrapping */
        .top-title {
          font-size: 2.0rem;
          font-weight: 800;
          line-height: 1.15;
          margin: 0;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    LOGO_PATH = "assets/milwaukee_logo.png"

    # --- Top header (logo + title) ---
    colA, colB = st.columns([1, 7], vertical_alignment="center")
    with colA:
        if os.path.exists(LOGO_PATH):
            render_logo(LOGO_PATH, sidebar=False, width=120)   # dùng if/else, không ternary
        else:
            st.caption("Missing logo: assets/milwaukee_logo.png")

    with colB:
        st.markdown(f"<div class='top-title'>{APP_TITLE}</div>", unsafe_allow_html=True)

    ensure_dirs()
    db_init()

    # --- Sidebar: logo nhỏ lại để không chiếm hết chỗ ---
    if os.path.exists(LOGO_PATH):
        render_logo(LOGO_PATH, sidebar=True, width=230)

    from streamlit_option_menu import option_menu

    # Menu luôn ở trên
    with st.sidebar:
        choice = option_menu(
            "Menu",
            ["Home (Upload)", "Dashboard", "History", "Help", "Admin"],
            icons=["cloud-upload", "bar-chart", "clock-history", "question-circle", "shield-lock"],
            menu_icon="list",
            default_index=0,
        )

    # Auth luôn là phần render CUỐI sidebar để bị đẩy xuống đáy bởi CSS last-child
    ensure_login()
    require_authorized_user()

    # ===== ROUTE PAGES (bạn đang thiếu đoạn này nên giao diện trống) =====
    if choice == "Home (Upload)":
        page_home_upload()
    elif choice == "Dashboard":
        page_dashboard()
    elif choice == "History":
        page_history()
    elif choice == "Help":
        page_help()
    elif choice == "Admin":
        # nếu muốn ẩn Admin với non-owner, giữ check ở đây
        if is_owner():
            page_admin()
        else:
            st.error("Bạn không có quyền truy cập Admin.")
    else:
        st.error("Unknown page")


if __name__ == "__main__":
    main()
