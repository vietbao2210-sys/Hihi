from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from openpyxl.formatting.rule import CellIsRule

import math
import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo

Bucket = Tuple[str, str]               # (zone, family)
Edge = Tuple[str, str, str, str]       # (from_zone, from_family, to_zone, to_family)


# ----------------------------
# Settings / Input structures
# ----------------------------

@dataclass(frozen=True)
class Settings:
    horizon_weeks: int = 8
    lead_time_weeks: int = 0
    integer_hiring: bool = True
    allow_within_zone_transfer: bool = True
    tie_break_transfers: bool = True
    hard_hire_gate: bool = False   # << NEW: ràng buộc cứng cho hiring
    epsilon: float = 1e-6
    big_m: Optional[float] = None



@dataclass
class InputData:
    settings: Settings
    weeks: List[int]
    buckets: List[Bucket]
    allowed_edges: List[Edge]

    A0: Dict[Bucket, float]
    GCOUNT: Dict[Bucket, int]

    PMC: Dict[Tuple[int, Bucket], float]
    GR: Dict[Tuple[int, Bucket], int]
    UR: Dict[Tuple[int, Bucket], float]

    PD: Dict[Tuple[int, Bucket], float]
    PD_per_group: Dict[Tuple[int, Bucket], float]

    TO: Dict[Tuple[int, Bucket], float]
    AB: Dict[Tuple[int, Bucket], float]
    ta_cap: Dict[int, float]

    zone_transfer_score: Dict[Tuple[str, str], int]


# ----------------------------
# Helpers
# ----------------------------

REQUIRED_BUCKET_COLS = ["zone", "family", "group_count", "op_per_group"]
REQUIRED_TA_COLS = ["week", "hiring_capacity"]


def _finite_max(vals, default: float = 0.0) -> float:
    m = default
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            m = max(m, fv)
    return m


def _finite_sum(vals) -> float:
    s = 0.0
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            s += fv
    return s


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, np.integer)):
        return int(x) != 0
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)):
            return False
        return float(x) != 0.0
    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "0.0", "false", "no", "n", "f", ""}:
        return False
    return True
def _infer_demand_time_axis(xl: pd.ExcelFile, sheet_name: str) -> Tuple[int, List[int]]:
    """
    Demand format: row 'Week' có các cột [baseline_week, week+1, ...].
    Return:
      base_week: tuần baseline (cột đầu tiên sau 'Week')
      plan_weeks: các tuần planning (các cột còn lại)
    """
    df = pd.read_excel(xl, sheet_name, header=None, dtype=object)

    week_sets = []
    for i in range(df.shape[0]):
        first = df.iat[i, 0]
        if isinstance(first, str) and _norm_label(first) == "week":
            ws = []
            for col in range(1, df.shape[1]):
                cell = df.iat[i, col]
                if _is_blank(cell):
                    continue
                try:
                    ws.append(_parse_week(cell))
                except Exception:
                    continue
            if ws:
                week_sets.append(sorted(set(ws)))

    if not week_sets:
        raise ValueError("Cannot find a 'Week' row in sheet 'demand'.")

    # optional: check all blocks use same header
    base = week_sets[0]
    for s in week_sets[1:]:
        if s != base:
            raise ValueError(f"Inconsistent week headers across demand blocks: {base} vs {s}")

    base_week = min(base)
    plan_weeks = sorted([w for w in base if w != base_week])
    return base_week, plan_weeks


def _list_buckets_from_demand(xl: pd.ExcelFile, sheet_name: str) -> List[Bucket]:
    """Lấy danh sách (zone,family) từ các dòng title 'Zone | Family' trong demand."""
    df = pd.read_excel(xl, sheet_name, header=None, dtype=object)
    bs: List[Bucket] = []
    for i in range(df.shape[0]):
        v0 = df.iat[i, 0]
        if isinstance(v0, str) and "|" in v0:
            zone, fam = [p.strip() for p in v0.strip().split("|", 1)]
            bs.append((zone, fam))
    if not bs:
        raise ValueError("No demand blocks found in sheet 'demand' (missing 'Zone | Family' titles).")
    return sorted(set(bs))


def _parse_week(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip().upper()
    import re
    m = re.match(r"WK\s*(\d+)", s)
    if m:
        return int(m.group(1))
    return int(float(s))


def _safe_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (float, np.floating)) and math.isnan(float(x)):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def _auto_big_m(data: InputData) -> float:
    max_pd = _finite_max(data.PD.values(), default=0.0)
    max_a0 = _finite_max(data.A0.values(), default=0.0)
    sum_cap = _finite_sum(data.ta_cap.get(w, 0.0) for w in data.weeks)
    M = max_pd + max_a0 + sum_cap + 10.0
    if not math.isfinite(M) or M < 0:
        M = 0.0
    return float(M)

REQUIRED_ZTM_ZONES = ["Zone 1", "Zone 2", "Zone 3", "LV5"]

def _clamp_int(x, lo: int, hi: int, default: int) -> int:
    v = _safe_int(x, default=default)
    if v < lo: return lo
    if v > hi: return hi
    return v

def _read_zone_transfer_matrix(
    xl: pd.ExcelFile,
    sheet_name: str,
    zones_expected: List[str] = None,
) -> Dict[Tuple[str, str], int]:
    """
    Expect format:
      first col: from\to (from_zone)
      headers: to_zones
      values: 1..4 (4 highest priority)
    Return dict[(from_zone,to_zone)] = score
    Missing pairs default = 1, diagonal default = 4
    """
    zones_expected = zones_expected or REQUIRED_ZTM_ZONES

    df = pd.read_excel(xl, sheet_name, dtype=object).dropna(how="all")
    if df.empty:
        sc = {}
        for zi in zones_expected:
            for zj in zones_expected:
                sc[(zi, zj)] = 1 if zi == zj else 4
        return sc

    # normalize headers
    df.columns = [str(c).strip() for c in df.columns]
    from_col = df.columns[0]
    to_cols = [c.strip() for c in df.columns[1:]]

    # build scores
    score: Dict[Tuple[str, str], int] = {}

    for _, r in df.iterrows():
        from_zone = str(r[from_col]).strip()
        if _is_blank(from_zone):
            continue
        for to_zone in to_cols:
            raw = r[to_zone] if to_zone in df.columns else None
            score[(from_zone, to_zone)] = _clamp_int(raw, 1, 4, default=4)

    # ensure defaults for expected zones
    for zi in zones_expected:
        for zj in zones_expected:
            score.setdefault((zi, zj), 1 if zi == zj else 4)

    # ensure diagonal exists but DON'T override user's value
    for z in zones_expected:
        score.setdefault((z, z), 1)

    return score

# ----------------------------
# Excel IO
# ----------------------------
def _is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (float, np.floating)) and math.isnan(float(x)):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _norm_label(x) -> str:
    # normalize label để match: "turn over rate", "turnover_rate", "Turnover Rate"...
    s = str(x).strip().lower()
    s = s.replace("%", "")
    s = s.replace("_", "")
    s = s.replace(" ", "")
    return s


def _to_unit_interval(x, default=0.0) -> float:
    """
    Parse percent / rate về [0,1].
    Hỗ trợ:
    - Excel % thường đọc ra 0.08
    - string "8%" -> 0.08
    - numeric 8 -> 0.08 (giả sử là percent)
    """
    if _is_blank(x):
        return float(default)
    try:
        if isinstance(x, str):
            s = x.strip().replace("%", "")
            if s == "":
                return float(default)
            v = float(s)
            # nếu user nhập 8 nghĩa là 8%
            if v > 1.0 + 1e-9:
                v = v / 100.0
        else:
            v = float(x)
            if v > 1.0 + 1e-9:
                v = v / 100.0

        if not math.isfinite(v):
            return float(default)
        return float(min(1.0, max(0.0, v)))
    except Exception:
        return float(default)


def _to_float(x, default=0.0) -> float:
    if _is_blank(x):
        return float(default)
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _read_demand_blocks(
    xl: pd.ExcelFile,
    sheet_name: str,
    weeks: List[int],                 # model weeks: 1..horizon
    bucket_list: List[Bucket],
    op_per_group_map: Dict[Bucket, float],
    base_week: int,
) -> Tuple[Dict[Bucket, float],
           Dict[Tuple[int, Bucket], float],
           Dict[Tuple[int, Bucket], int],
           Dict[Tuple[int, Bucket], float],
           Dict[Tuple[int, Bucket], float],
           Dict[Tuple[int, Bucket], float]]:
    """
    Parse sheet demand dạng block:
    [Row Title]  "Zone 1 | Clean"
    [Row Week]   week  0 1 2 ... 8
    [Rows data]  pmc_demand / group_demand / ur_rate / Actual / turnover_rate / absent_rate

    Return:
      A0, PMC, GR, UR, TO, AB  cho t in weeks (1..horizon), và A0 từ Actual week 0
    """
    df = pd.read_excel(xl, sheet_name, header=None, dtype=object)

    nrows, ncols = df.shape
    bucket_set = set(bucket_list)

    A0: Dict[Bucket, float] = {}
    PMC: Dict[Tuple[int, Bucket], float] = {}
    GR: Dict[Tuple[int, Bucket], int] = {}
    UR: Dict[Tuple[int, Bucket], float] = {}
    TO: Dict[Tuple[int, Bucket], float] = {}
    AB: Dict[Tuple[int, Bucket], float] = {}

    i = 0
    while i < nrows:
        v0 = df.iat[i, 0]

        # start of block: "Zone X | Family"
        if isinstance(v0, str) and "|" in v0:
            title = v0.strip()
            parts = [p.strip() for p in title.split("|", 1)]
            if len(parts) < 2:
                i += 1
                continue

            zone, fam = parts[0], parts[1]
            b = (zone, fam)

            # nếu demand có GF không nằm trong buckets -> báo lỗi sớm (để bạn biết thiếu mapping)
            if b not in bucket_set:
                raise ValueError(f"Demand contains GF not found in 'buckets': {zone} | {fam}")

            # tìm row week
            i += 1
            while i < nrows:
                first = df.iat[i, 0]
                if isinstance(first, str) and _norm_label(first) in {"week"}:
                    break
                i += 1
            if i >= nrows:
                break

            # parse week columns map
            week_to_col: Dict[int, int] = {}
            for col in range(1, ncols):
                cell = df.iat[i, col]
                if _is_blank(cell):
                    continue
                try:
                    w = _parse_week(cell)
                    week_to_col[int(w)] = col
                except Exception:
                    continue

            i += 1  # move to first data row

            # collect rows inside this block until blank line or next title
            block: Dict[str, Dict[int, object]] = {}
            while i < nrows:
                first = df.iat[i, 0]

                # next block begins
                if isinstance(first, str) and "|" in first:
                    break

                # blank row => kết thúc block
                if _is_blank(first):
                    i += 1
                    break

                key = _norm_label(first)
                row_map: Dict[int, object] = {}
                for w, col in week_to_col.items():
                    row_map[w] = df.iat[i, col]
                block[key] = row_map
                i += 1

            # ---- map labels -> canonical keys
            def _get_row(*candidates: str) -> Dict[int, object]:
                for c in candidates:
                    if c in block:
                        return block[c]
                return {}

            pmc_row = _get_row("pmcdemand", "pmc")
            grp_row = _get_row("groupdemand", "group")
            ur_row  = _get_row("urrate", "ur", "underouting", "underroute")
            act_row = _get_row("actual", "hc0", "headcount")
            to_row  = _get_row("turnoverrate", "turnover", "turnoverr")
            ab_row  = _get_row("absentrate", "absent", "absence")

            # ---- A0 = Actual tại baseline week (cột B)
            a0_raw = act_row.get(base_week, None)
            if _is_blank(a0_raw):
                raise ValueError(f"Missing Actual (baseline week {base_week}) for GF: {zone} | {fam}")
            A0[b] = _to_float(a0_raw, default=0.0)

            # helper: fill rate theo tuần: nếu thiếu week t thì lấy gần nhất <=t, nếu vẫn thiếu thì lấy week 0, else 0
            def _filled_rate(row: Dict[int, object], t: int) -> float:
                if not row:
                    return 0.0
                if not _is_blank(row.get(t, None)):
                    return _to_unit_interval(row[t], default=0.0)

                cand = [w for w in row.keys() if w <= t and not _is_blank(row.get(w, None))]
                if cand:
                    return _to_unit_interval(row[max(cand)], default=0.0)

                # fallback baseline_week nếu có
                if not _is_blank(row.get(base_week, None)):
                    return _to_unit_interval(row[base_week], default=0.0)

                cand2 = [w for w in row.keys() if not _is_blank(row.get(w, None))]
                if cand2:
                    return _to_unit_interval(row[min(cand2)], default=0.0)

                return 0.0

            # detect if group_demand is totally empty -> optional auto-derive from op_per_group
            grp_has_any = any(not _is_blank(grp_row.get(t, None)) for t in weeks)

            oppg = float(op_per_group_map.get(b, 0.0))
            for t in weeks:
                pmc = _to_float(pmc_row.get(t, None), default=0.0)
                PMC[(t, b)] = pmc

                # group_demand: dùng row nếu có; nếu trống toàn bộ thì derive (tuỳ bạn có muốn)
                if grp_has_any:
                    GR[(t, b)] = max(0, _safe_int(grp_row.get(t, None), default=0))
                else:
                    # derive (report-only): ceil(pmc / op_per_group)
                    if oppg > 0 and pmc > 0:
                        GR[(t, b)] = int(math.ceil(pmc / oppg))
                    else:
                        GR[(t, b)] = 0

                UR[(t, b)] = _filled_rate(ur_row, t)
                TO[(t, b)] = _filled_rate(to_row, t)
                AB[(t, b)] = _filled_rate(ab_row, t)

        else:
            i += 1

    # sanity: ensure all buckets have A0
    missing = [b for b in bucket_list if b not in A0]
    if missing:
        msg = "\n".join([f"- {z} | {f}" for (z, f) in missing[:30]])
        raise ValueError(f"Missing demand blocks (Actual baseline week {base_week}) for some buckets:\n{msg}")

    return A0, PMC, GR, UR, TO, AB

def read_input_excel(path: str) -> InputData:
    xl = pd.ExcelFile(path)
    sheet_lc = [s.lower() for s in xl.sheet_names]

    # ---- settings
    if "settings" in sheet_lc:
        real = xl.sheet_names[sheet_lc.index("settings")]
        s_df = pd.read_excel(xl, real).dropna(how="all")
        s_df.columns = [c.strip().lower() for c in s_df.columns]

        kv = {}
        if "key" in s_df.columns and "value" in s_df.columns:
            for _, r in s_df.iterrows():
                k = str(r["key"]).strip().lower()
                kv[k] = r["value"]

        eps_raw = kv.get("epsilon", 1e-6)
        try:
            eps_val = float(eps_raw)
            if not math.isfinite(eps_val):
                eps_val = 1e-6
        except Exception:
            eps_val = 1e-6

        big_m_val = None
        if "big_m" in kv:
            try:
                tmp = float(kv["big_m"])
                if math.isfinite(tmp) and tmp >= 0:
                    big_m_val = tmp
            except Exception:
                big_m_val = None

        settings = Settings(
            horizon_weeks=int(kv.get("horizon_weeks", 8)),
            lead_time_weeks=int(kv.get("lead_time_weeks", 0)),
            integer_hiring=_to_bool(kv.get("integer_hiring", True)),
            allow_within_zone_transfer=_to_bool(kv.get("allow_within_zone_transfer", True)),
            tie_break_transfers=_to_bool(kv.get("tie_break_transfers", True)),
            hard_hire_gate=_to_bool(kv.get("hard_hire_gate", False)),  # << NEW
            epsilon=eps_val,
            big_m=big_m_val,
        )

    else:
        settings = Settings()

    # ---- demand axis (baseline + planning weeks)
    real_demand = xl.sheet_names[sheet_lc.index("demand")]
    base_week, plan_weeks_all = _infer_demand_time_axis(xl, real_demand)

    # settings.horizon_weeks = số tuần planning (không tính baseline)
    if settings.horizon_weeks > len(plan_weeks_all):
        raise ValueError(
            f"Demand only has {len(plan_weeks_all)} planning weeks after baseline {base_week}, "
            f"but horizon_weeks={settings.horizon_weeks}."
        )

    weeks = plan_weeks_all[:settings.horizon_weeks]

    # ---- buckets
    if "buckets" not in sheet_lc:
        raise ValueError("Missing required sheet: 'buckets'")
    real = xl.sheet_names[sheet_lc.index("buckets")]
    buckets_df = pd.read_excel(xl, real).dropna(how="all")
    buckets_df.columns = [c.strip().lower() for c in buckets_df.columns]

    for c in REQUIRED_BUCKET_COLS:
        if c not in buckets_df.columns:
            raise ValueError(f"Sheet 'buckets' missing required column: {c}")

    buckets_df["zone"] = buckets_df["zone"].astype(str).str.strip()
    buckets_df["family"] = buckets_df["family"].astype(str).str.strip()
    buckets_df = buckets_df.drop_duplicates(subset=["zone", "family"], keep="first").reset_index(drop=True)

    bucket_list: List[Bucket] = sorted([(r.zone, r.family) for r in buckets_df.itertuples(index=False)])
    if not bucket_list:
        raise ValueError("Sheet 'buckets' is empty after cleaning.")

    # fixed group_count (>=1)
    GCOUNT: Dict[Bucket, int] = {}
    OPPG: Dict[Bucket, float] = {}
    for r in buckets_df.itertuples(index=False):
        b = (str(r.zone), str(r.family))
        gc = _safe_int(getattr(r, "group_count", 0), default=0)
        GCOUNT[b] = max(1, gc)
        OPPG[b] = _to_float(getattr(r, "op_per_group", 0.0), default=0.0)

    # NOTE: A0 / TO / AB sẽ lấy từ sheet demand (không còn lấy từ buckets)

    # ---- demand (NEW: block format)
    if "demand" not in sheet_lc:
        raise ValueError("Missing required sheet: 'demand'")
    real = xl.sheet_names[sheet_lc.index("demand")]

    A0, PMC, GR, UR, TO, AB = _read_demand_blocks(
        xl=xl,
        sheet_name=real,
        weeks=weeks,
        bucket_list=bucket_list,
        op_per_group_map=OPPG,
        base_week=base_week,
    )

    # ---- compute PD and PD_per_group
    PD: Dict[Tuple[int, Bucket], float] = {}
    PDpg: Dict[Tuple[int, Bucket], float] = {}

    for t in weeks:
        for b in bucket_list:
            pmc = float(PMC.get((t, b), 0.0))
            ur = float(UR.get((t, b), 0.0))
            prod_demand = pmc * (1.0 - ur)

            PD[(t, b)] = prod_demand
            gr_int = int(GR.get((t, b), 0))
            PDpg[(t, b)] = (prod_demand / float(gr_int)) if gr_int > 0 else 0.0


    # ---- TA capacity
    if "ta_capacity" not in sheet_lc:
        raise ValueError("Missing required sheet: 'ta_capacity'")
    real = xl.sheet_names[sheet_lc.index("ta_capacity")]
    ta_df = pd.read_excel(xl, real).dropna(how="all")
    ta_df.columns = [c.strip().lower() for c in ta_df.columns]
    for c in REQUIRED_TA_COLS:
        if c not in ta_df.columns:
            raise ValueError(f"Sheet 'ta_capacity' missing required column: {c}")
    ta_df["week"] = ta_df["week"].apply(_parse_week)
    ta_df = ta_df[ta_df["week"].isin(weeks)].copy()

    ta_cap = {int(r.week): float(r.hiring_capacity) for r in ta_df.itertuples(index=False)}
    for w in weeks:
        ta_cap.setdefault(w, 0.0)


    # ---- allowed transfer edges
    def allow_all_pairs() -> List[Edge]:
        ed: List[Edge] = []
        for (zi, fi) in bucket_list:
            for (zj, fj) in bucket_list:
                if (zi, fi) == (zj, fj):
                    continue
                if not settings.allow_within_zone_transfer and zi == zj:
                    continue
                ed.append((zi, fi, zj, fj))
        return ed

    allowed_edges: List[Edge] = []
    if "transfer_allowed" in sheet_lc:
        real = xl.sheet_names[sheet_lc.index("transfer_allowed")]
        tr = pd.read_excel(xl, real).dropna(how="all")
        if tr.empty:
            allowed_edges = allow_all_pairs()
        else:
            tr.columns = [c.strip().lower() for c in tr.columns]
            req = ["from_zone", "from_family", "to_zone", "to_family", "allowed"]
            for c in req:
                if c not in tr.columns:
                    raise ValueError(f"Sheet 'transfer_allowed' missing required column: {c}")
            tr["from_zone"] = tr["from_zone"].astype(str).str.strip()
            tr["from_family"] = tr["from_family"].astype(str).str.strip()
            tr["to_zone"] = tr["to_zone"].astype(str).str.strip()
            tr["to_family"] = tr["to_family"].astype(str).str.strip()
            tr["allowed"] = tr["allowed"].apply(_to_bool)

            allowed_set = {(r.from_zone, r.from_family, r.to_zone, r.to_family)
                           for r in tr.itertuples(index=False) if r.allowed}

            for (zi, fi) in bucket_list:
                for (zj, fj) in bucket_list:
                    if (zi, fi) == (zj, fj):
                        continue
                    if not settings.allow_within_zone_transfer and zi == zj:
                        continue
                    if (zi, fi, zj, fj) in allowed_set:
                        allowed_edges.append((zi, fi, zj, fj))
    else:
        allowed_edges = allow_all_pairs()
    zone_score: Dict[Tuple[str, str], int] = {}

    if "zone_transfer_matrix" in sheet_lc:
        real = xl.sheet_names[sheet_lc.index("zone_transfer_matrix")]
        zone_score = _read_zone_transfer_matrix(xl, real, zones_expected=REQUIRED_ZTM_ZONES)
    else:
        # fallback nếu sheet chưa có
        zone_score = {(zi, zj): (1 if zi == zj else 4)
                      for zi in REQUIRED_ZTM_ZONES for zj in REQUIRED_ZTM_ZONES}
    return InputData(
        settings=settings,
        weeks=weeks,
        buckets=bucket_list,
        allowed_edges=allowed_edges,
        A0=A0,
        GCOUNT=GCOUNT,
        PMC=PMC,
        GR=GR,
        UR=UR,
        PD=PD,
        PD_per_group=PDpg,
        TO=TO,
        AB=AB,
        ta_cap=ta_cap,
        zone_transfer_score=zone_score,
    )


# ----------------------------
# Pyomo Model
# ----------------------------

def build_model(data: InputData) -> pyo.ConcreteModel:
    s = data.settings
    weeks = data.weeks
    buckets = data.buckets
    edges = data.allowed_edges

    M = s.big_m if s.big_m is not None else _auto_big_m(data)
    if not (isinstance(M, (int, float)) and math.isfinite(float(M)) and float(M) >= 0):
        raise ValueError(f"BigM invalid ({M}). Check input or set a finite big_m in 'settings'.")

    # adjacency
    out_edges: Dict[Bucket, List[Edge]] = {b: [] for b in buckets}
    in_edges: Dict[Bucket, List[Edge]] = {b: [] for b in buckets}
    for e in edges:
        out_edges[(e[0], e[1])].append(e)
        in_edges[(e[2], e[3])].append(e)

    m = pyo.ConcreteModel("Phase1_Planning_v4")
    m.T = pyo.Set(initialize=weeks, ordered=True)
    m.B = pyo.Set(initialize=buckets, dimen=2)
    m.E = pyo.Set(initialize=edges, dimen=4)

    # NEW: score per edge, mapped by (from_zone,to_zone)
    def ScoreE_init(mm, zi, fi, zj, fj):
        return int(data.zone_transfer_score.get((str(zi), str(zj)), 4))

    m.ScoreE = pyo.Param(m.E, initialize=ScoreE_init, within=pyo.NonNegativeIntegers)

    m.BigM = pyo.Param(initialize=float(M), within=pyo.NonNegativeReals, mutable=False)

    # Params
    def init_dict(dct, default=0.0):
        def _rule(mm, t, z, f):
            return float(dct.get((int(t), (str(z), str(f))), default))
        return _rule

    def init_dict_int(dct, default=0):
        def _rule(mm, t, z, f):
            return int(dct.get((int(t), (str(z), str(f))), default))
        return _rule

    m.PMC = pyo.Param(m.T, m.B, initialize=init_dict(data.PMC), within=pyo.NonNegativeReals)
    m.GR  = pyo.Param(m.T, m.B, initialize=init_dict_int(data.GR), within=pyo.NonNegativeIntegers)
    m.UR  = pyo.Param(m.T, m.B, initialize=init_dict(data.UR, default=0.0), within=pyo.UnitInterval)

    m.PD  = pyo.Param(m.T, m.B, initialize=init_dict(data.PD), within=pyo.NonNegativeReals)       # required
    m.PDpg = pyo.Param(m.T, m.B, initialize=init_dict(data.PD_per_group), within=pyo.NonNegativeReals)

    m.TO = pyo.Param(m.T, m.B, initialize=init_dict(data.TO), within=pyo.UnitInterval)
    m.AB = pyo.Param(m.T, m.B, initialize=init_dict(data.AB), within=pyo.UnitInterval)

    def A0_init(mm, z, f):
        return float(data.A0[(str(z), str(f))])
    m.A0 = pyo.Param(m.B, initialize=A0_init, within=pyo.NonNegativeReals)

    def GC_init(mm, z, f):
        return int(data.GCOUNT.get((str(z), str(f)), 1))
    m.GC = pyo.Param(m.B, initialize=GC_init, within=pyo.NonNegativeIntegers)

    def Cap_init(mm, t):
        return float(data.ta_cap.get(int(t), 0.0))
    m.Cap = pyo.Param(m.T, initialize=Cap_init, within=pyo.NonNegativeReals)

    # Vars
    hire_domain = pyo.NonNegativeIntegers if s.integer_hiring else pyo.NonNegativeReals
    m.HR = pyo.Var(m.T, m.B, domain=hire_domain)
    # --- Tie-break objective: hire càng trễ càng tốt
    Tmax = max(weeks)

    m.ObjLateHire = pyo.Objective(
        expr=sum((Tmax - int(t)) * sum(m.HR[t, b] for b in m.B) for t in m.T),
        sense=pyo.minimize
    )
    m.ObjLateHire.deactivate()

    m.A  = pyo.Var(m.T, m.B, domain=pyo.NonNegativeReals)
    m.N  = pyo.Var(m.T, m.B, domain=pyo.NonNegativeReals)

    # slack (HC shortage only)
    m.SHC = pyo.Var(m.T, m.B, domain=pyo.NonNegativeReals)

    # transfers
    m.X = pyo.Var(m.T, m.E, domain=pyo.NonNegativeReals)

    m.TotalTransferScore = pyo.Expression(
        expr=sum(m.X[t, e] * m.ScoreE[e] for t in m.T for e in m.E)
    )

    # NEW: objective to MINIMIZE total score
    m.ObjScore = pyo.Objective(expr=m.TotalTransferScore, sense=pyo.minimize)
    m.ObjScore.deactivate()

    m.can_lend = pyo.Var(m.T, m.B, domain=pyo.Binary)
    m.can_borrow = pyo.Var(m.T, m.B, domain=pyo.Binary)
    m.GapPre = pyo.Expression(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.N[t, (str(z), str(f))] - mm.PD[t, (str(z), str(f))]
    )
    # -----------------
    # Overstaff slack (dư trước transfer) = max(0, GapPre)
    # -----------------
    m.OHC = pyo.Var(m.T, m.B, domain=pyo.NonNegativeReals)

    # OHC >= GapPre  (nếu GapPre âm thì OHC tự về 0)
    m.C_over = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.OHC[t, (str(z), str(f))] >= mm.GapPre[t, (str(z), str(f))]
    )

    # Objective: minimize total overstaff (để dùng ở stage riêng)
    m.ObjOver = pyo.Objective(
        expr=sum(m.OHC[t, b] for t in m.T for b in m.B),
        sense=pyo.minimize
    )
    m.ObjOver.deactivate()

    # Expressions Out/In and Net after transfer
    def out_expr(mm, t, z, f):
        b = (str(z), str(f))
        return sum(mm.X[t, e] for e in out_edges[b])

    def in_expr(mm, t, z, f):
        b = (str(z), str(f))
        return sum(mm.X[t, e] for e in in_edges[b])

    m.Out = pyo.Expression(m.T, m.B, rule=out_expr)
    m.In  = pyo.Expression(m.T, m.B, rule=in_expr)

    m.NetAfter = pyo.Expression(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.N[t, (str(z), str(f))] - mm.Out[t, (str(z), str(f))] + mm.In[t, (str(z), str(f))]
    )

    # -----------------
    # Constraints
    # -----------------

    # (C2) Stock-flow with turnover + lead time
    L = int(s.lead_time_weeks)
    min_t = min(weeks)

    def c2_rule(mm, t, z, f):
        t = int(t)
        b = (str(z), str(f))
        prev_A = mm.A0[b] if t == min_t else mm.A[t - 1, b]
        carried = prev_A * (1 - mm.TO[t, b])

        if L <= 0:
            inflow_hire = mm.HR[t, b]
        else:
            inflow_hire = mm.HR[t - L, b] if (t - L) in weeks else 0.0

        return mm.A[t, b] == carried + inflow_hire

    m.C2 = pyo.Constraint(m.T, m.B, rule=c2_rule)

    # (C3) Net after absenteeism
    m.C3 = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.N[t, (str(z), str(f))] == mm.A[t, (str(z), str(f))] * (1 - mm.AB[t, (str(z), str(f))])
    )
    # ---- Pre-hire (lead_time=0) baseline: carried headcount only (không tính HR[t,b])
    L = int(s.lead_time_weeks)
    min_t = min(weeks)

    if s.hard_hire_gate and L != 0:
        raise ValueError("hard_hire_gate currently supports lead_time_weeks=0 only.")

    def carried_A_rule(mm, t, z, f):
        t = int(t)
        b = (str(z), str(f))
        prev_A = mm.A0[b] if t == min_t else mm.A[t - 1, b]
        return prev_A * (1 - mm.TO[t, b])

    m.A_carried = pyo.Expression(m.T, m.B, rule=carried_A_rule)

    # N_before_hire = carried_A * (1-AB)  (chưa cộng HR[t,b])
    m.N_before_hire = pyo.Expression(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.A_carried[t, (str(z), str(f))] * (1 - mm.AB[t, (str(z), str(f))])
    )

    # Gap_before_hire = N_before_hire - PD
    m.GapBeforeHire = pyo.Expression(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.N_before_hire[t, (str(z), str(f))] - mm.PD[t, (str(z), str(f))]
    )
    # Binary: 1 nếu bucket đang thiếu (GapBeforeHire < 0) -> được phép tuyển
    m.need_hire = pyo.Var(m.T, m.B, domain=pyo.Binary)
    # =========================
    # Lookahead pre-hire gate (lead_time=0)
    # =========================
    T_max = max(weeks)

    # set weeks excluding last
    m.T_pre = pyo.Set(initialize=[t for t in weeks if t < T_max], ordered=True)

    # Deficit before hire per bucket: DefBH[t,b] = max(0, -GapBeforeHire[t,b])
    m.DefBH = pyo.Var(m.T, m.B, domain=pyo.NonNegativeReals)

    m.C_defbh_ge = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.DefBH[t, (str(z), str(f))] >= -mm.GapBeforeHire[t, (str(z), str(f))]
    )
    m.C_defbh_le1 = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.DefBH[t, (str(z), str(f))] <= mm.BigM * mm.need_hire[t, (str(z), str(f))]
    )
    m.C_defbh_le2 = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.DefBH[t, (str(z), str(f))] <= -mm.GapBeforeHire[t, (str(z), str(f))] + mm.BigM * (
                    1 - mm.need_hire[t, (str(z), str(f))])
    )

    # Total deficit before hire per week
    m.TotalDefBH = pyo.Expression(m.T, rule=lambda mm, t: sum(mm.DefBH[t, b] for b in mm.B))

    # A bigger M for weekly sums
    m.BigMTotal = pyo.Param(initialize=float(M) * max(1, len(buckets)), within=pyo.NonNegativeReals)

    # OverflowNext[t] = 1 if TotalDefBH[t+1] > Cap[t+1]
    m.OverflowNext = pyo.Var(m.T_pre, domain=pyo.Binary)

    m.C_overflow_lb = pyo.Constraint(
        m.T_pre,
        rule=lambda mm, t: (mm.TotalDefBH[t + 1] - mm.Cap[t + 1]) >= s.epsilon - mm.BigMTotal * (1 - mm.OverflowNext[t])
    )
    m.C_overflow_ub = pyo.Constraint(
        m.T_pre,
        rule=lambda mm, t: (mm.TotalDefBH[t + 1] - mm.Cap[t + 1]) <= 0.0 + mm.BigMTotal * mm.OverflowNext[t]
    )

    # prehire_ok[t,b] = OverflowNext[t] AND need_hire[t+1,b]
    m.prehire_ok = pyo.Var(m.T_pre, m.B, domain=pyo.Binary)

    m.C_prehire1 = pyo.Constraint(m.T_pre, m.B,
                                  rule=lambda mm, t, z, f: mm.prehire_ok[t, (str(z), str(f))] <= mm.OverflowNext[t])
    m.C_prehire2 = pyo.Constraint(m.T_pre, m.B,
                                  rule=lambda mm, t, z, f: mm.prehire_ok[t, (str(z), str(f))] <= mm.need_hire[
                                      t + 1, (str(z), str(f))])
    m.C_prehire3 = pyo.Constraint(
        m.T_pre, m.B,
        rule=lambda mm, t, z, f: mm.prehire_ok[t, (str(z), str(f))] >= mm.OverflowNext[t] + mm.need_hire[
            t + 1, (str(z), str(f))] - 1
    )

    # Hiring allowed:
    # - if thiếu tuần hiện tại: need_hire[t,b] = 1
    # - OR nếu tuần sau overflow cap & bucket tuần sau thiếu: prehire_ok[t,b] = 1
    m.C_hire_gate_now_or_prehire = pyo.Constraint(
        m.T_pre, m.B,
        rule=lambda mm, t, z, f:
        mm.HR[t, (str(z), str(f))] <= mm.BigM * mm.need_hire[t, (str(z), str(f))] + mm.BigM * mm.prehire_ok[
            t, (str(z), str(f))]
    )

    # Last week: chỉ cho hire nếu thiếu tuần đó
    m.C_hire_gate_last = pyo.Constraint(
        pyo.Set(initialize=[T_max]), m.B,
        rule=lambda mm, t, z, f:
        mm.HR[t, (str(z), str(f))] <= mm.BigM * mm.need_hire[t, (str(z), str(f))]
    )

    # Nếu need_hire = 0 => GapBeforeHire >= 0  (không thiếu)
    m.C_hire_gate_lb = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f:
        mm.GapBeforeHire[t, (str(z), str(f))] >= 0.0 - mm.BigM * mm.need_hire[t, (str(z), str(f))]
    )

    # Nếu need_hire = 1 => GapBeforeHire <= -epsilon  (đang thiếu thật)
    m.C_hire_gate_ub = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f:
        mm.GapBeforeHire[t, (str(z), str(f))] <= -s.epsilon + mm.BigM * (1 - mm.need_hire[t, (str(z), str(f))])
    )


    # (C4) TA capacity per week
    m.C4 = pyo.Constraint(m.T, rule=lambda mm, t: sum(mm.HR[t, b] for b in mm.B) <= mm.Cap[t])

    # (C5) Meet production demand after transfers + slack
    m.C5 = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.NetAfter[t, (str(z), str(f))] + mm.SHC[t, (str(z), str(f))] >= mm.PD[t, (str(z), str(f))]
    )

    # -------- Transfer direction logic (same as v1; based on REQUIRED PD) --------
    # gap_pre = N - PD (before transfer)
    # Không được vừa lend vừa borrow
    m.C60 = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.can_lend[t, (str(z), str(f))] + mm.can_borrow[t, (str(z), str(f))] <= 1
    )

    # Nếu can_lend = 1 => GAP >= 0.5
    m.C61_lend_th = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.GapPre[t, (str(z), str(f))] >= 0.5 - mm.BigM * (1 - mm.can_lend[t, (str(z), str(f))])
    )

    # Nếu can_borrow = 1 => GAP <= -0.5
    m.C61_borrow_th = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.GapPre[t, (str(z), str(f))] <= -0.5 + mm.BigM * (
                    1 - mm.can_borrow[t, (str(z), str(f))])
    )

    # Nếu không bật can_lend/can_borrow thì Out/In phải = 0
    m.C62_out_active = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.Out[t, (str(z), str(f))] <= mm.BigM * mm.can_lend[t, (str(z), str(f))]
    )
    m.C63_in_active = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.In[t, (str(z), str(f))] <= mm.BigM * mm.can_borrow[t, (str(z), str(f))]
    )

    # Không cho lend vượt quá surplus, và borrow vượt quá deficit (tightening tốt cho solver)
    m.C64_out_limit = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.Out[t, (str(z), str(f))] <= mm.GapPre[t, (str(z), str(f))] + mm.BigM * (
                    1 - mm.can_lend[t, (str(z), str(f))])
    )
    m.C64_in_limit = pyo.Constraint(
        m.T, m.B,
        rule=lambda mm, t, z, f: mm.In[t, (str(z), str(f))] <= (-mm.GapPre[t, (str(z), str(f))]) + mm.BigM * (
                    1 - mm.can_borrow[t, (str(z), str(f))])
    )

    # Objective stage 1: minimize HC shortage only
    m.ObjSlack = pyo.Objective(
        expr=sum(m.SHC[t, b] for t in m.T for b in m.B),
        sense=pyo.minimize
    )

    return m


# ----------------------------
# Solve (lexicographic)
# ----------------------------

@dataclass
class SolveResult:
    status: str
    termination_condition: str
    total_shortage: float
    total_hiring: float
    total_transfer: float
    total_transfer_score: float


def _get_highs_solver():
    try:
        from pyomo.contrib.appsi.solvers import Highs
        return Highs(), "appsi"
    except Exception:
        solver = pyo.SolverFactory("highs")
        return solver, "legacy"


def solve_model(
    m: pyo.ConcreteModel,
    tie_break_transfers: bool = True,   # <-- bật/tắt bằng settings trong Excel
    epsilon: float = 1e-6,
) -> SolveResult:
    solver, mode = _get_highs_solver()

    def _solve():
        if mode == "appsi":
            res = solver.solve(m)
            term = str(getattr(res, "termination_condition", "unknown"))
            status = "ok"
        else:
            res = solver.solve(m, tee=False)
            term = str(res.solver.termination_condition)
            status = str(res.solver.status)
        return status, term

    # ----------------------------
    # Stage 1: min shortage
    # ----------------------------
    m.ObjSlack.activate()
    if hasattr(m, "ObjHire"): m.ObjHire.deactivate()
    if hasattr(m, "ObjOver"): m.ObjOver.deactivate()
    if hasattr(m, "ObjMove"): m.ObjMove.deactivate()

    status, term = _solve()
    total_shortage = float(pyo.value(m.ObjSlack))

    if hasattr(m, "FixShortage"):
        m.del_component(m.FixShortage)
    m.FixShortage = pyo.Constraint(
        expr=sum(m.SHC[t, b] for t in m.T for b in m.B) <= total_shortage + epsilon
    )

    # ----------------------------
    # Stage 2: min total hiring
    # ----------------------------
    if hasattr(m, "ObjHire"):
        m.ObjHire.activate()
    else:
        m.ObjHire = pyo.Objective(
            expr=sum(m.HR[t, b] for t in m.T for b in m.B),
            sense=pyo.minimize
        )

    m.ObjSlack.deactivate()
    if hasattr(m, "ObjOver"): m.ObjOver.deactivate()
    if hasattr(m, "ObjMove"): m.ObjMove.deactivate()

    status, term = _solve()
    total_hiring = float(pyo.value(m.ObjHire))

    if hasattr(m, "FixHire"):
        m.del_component(m.FixHire)
    m.FixHire = pyo.Constraint(
        expr=sum(m.HR[t, b] for t in m.T for b in m.B) <= total_hiring + epsilon
    )

    # ----------------------------
    # ----------------------------
    # Stage 3: hire càng trễ càng tốt (tie-break)
    # ----------------------------
    m.ObjLateHire.activate()
    m.ObjHire.deactivate()
    m.ObjSlack.deactivate()
    if hasattr(m, "ObjOver"): m.ObjOver.deactivate()
    if hasattr(m, "ObjMove"): m.ObjMove.deactivate()

    status, term = _solve()
    late_val = float(pyo.value(m.ObjLateHire))

    if hasattr(m, "FixLate"):
        m.del_component(m.FixLate)
    m.FixLate = pyo.Constraint(expr=m.ObjLateHire.expr <= late_val + epsilon)

    m.ObjLateHire.deactivate()

    total_transfer_score = 0.0
    total_transfer = 0.0

    # ----------------------------
    # Stage 4: MIN transfer score  (score thắng min transfer)
    # ----------------------------
    if tie_break_transfers:
        if not hasattr(m, "ObjScore"):
            raise ValueError("Model missing ObjScore. Check build_model() placement.")

        # deactivate objectives khác
        m.ObjSlack.deactivate()
        if hasattr(m, "ObjHire"): m.ObjHire.deactivate()
        if hasattr(m, "ObjOver"): m.ObjOver.deactivate()
        if hasattr(m, "ObjLateHire"): m.ObjLateHire.deactivate()
        if hasattr(m, "ObjMove"): m.ObjMove.deactivate()

        # MIN score
        m.ObjScore.activate()
        status, term = _solve()
        total_transfer_score = float(pyo.value(m.TotalTransferScore))

        # Fix score at optimum (minimize => <=)
        if hasattr(m, "FixScore"):
            m.del_component(m.FixScore)
        m.FixScore = pyo.Constraint(expr=m.TotalTransferScore <= total_transfer_score + epsilon)

        m.ObjScore.deactivate()

        # ----------------------------
        # Stage 5: MIN transfer given min score
        # ----------------------------
        if hasattr(m, "ObjMove"):
            m.del_component(m.ObjMove)
        m.ObjMove = pyo.Objective(
            expr=sum(m.X[t, e] for t in m.T for e in m.E),
            sense=pyo.minimize
        )

        # deactivate objectives khác
        m.ObjSlack.deactivate()
        if hasattr(m, "ObjHire"): m.ObjHire.deactivate()
        if hasattr(m, "ObjOver"): m.ObjOver.deactivate()
        if hasattr(m, "ObjLateHire"): m.ObjLateHire.deactivate()

        m.ObjMove.activate()
        status, term = _solve()
        total_transfer = float(pyo.value(m.ObjMove))
        m.ObjMove.deactivate()

    return SolveResult(
        status=status,
        termination_condition=term,
        total_shortage=total_shortage,
        total_hiring=total_hiring,
        total_transfer=total_transfer,
        total_transfer_score=total_transfer_score,
    )




# ----------------------------
# Output
# ----------------------------

def extract_results(data: InputData, m: pyo.ConcreteModel) -> Dict[str, pd.DataFrame]:
    T = list(m.T)
    B = list(m.B)
    min_t = min(T)

    rows = []
    for t in T:
        for (z, f) in B:
            b = (str(z), str(f))
            t_int = int(t)

            # ---- inputs / params
            pmc = float(pyo.value(m.PMC[t, b]))
            ur_rate = float(pyo.value(m.UR[t, b]))          # 0..1
            to_rate = float(pyo.value(m.TO[t, b]))          # 0..1
            ab_rate = float(pyo.value(m.AB[t, b]))          # 0..1

            gr = int(pyo.value(m.GR[t, b]))
            gc = int(pyo.value(m.GC[b]))

            # ---- demand
            pd_prod = float(pyo.value(m.PD[t, b]))          # Production Demand = PMC*(1-UR)
            ppl_per_group = (pd_prod / gr) if gr > 0 else 0.0

            # ---- workforce results
            actual = float(pyo.value(m.A[t, b]))            # after turnover + hires-inflow
            net_pre = float(pyo.value(m.N[t, b]))  # CHƯA tính transfer
            hire = float(pyo.value(m.HR[t, b]))

            # --- raw (để tính toán, KHÔNG làm tròn)
            infl_raw = float(pyo.value(m.In[t, b]))
            outf_raw = float(pyo.value(m.Out[t, b]))
            net_after = net_pre - outf_raw + infl_raw

            # --- display only (để report, có làm tròn)
            infl_disp = int(round(infl_raw))
            outf_disp = int(round(outf_raw))

            # --- Bucket Weekly: giữ raw, nhưng nếu đúng 0 thì để trống (tuỳ bạn)
            eps0 = 1e-12
            infl_cell = None if abs(infl_raw) < eps0 else infl_raw
            outf_cell = None if abs(outf_raw) < eps0 else outf_raw

            gap_pre = net_pre - pd_prod  # CHƯA transfer
            gap_after = net_after - pd_prod  # ĐÃ transfer

            hc_net_before_hiring_after = net_after - hire  # theo “HC net (đã transfer) - hire”
            hc_net_before_hiring_pre = net_pre - hire  # theo “HC net (chưa transfer) - hire”

            shc = float(pyo.value(m.SHC[t, b]))


            rows.append({
                "Week": t_int,
                "Zone": b[0],
                "Family": b[1],

                "GF": f"{b[0]} | {b[1]}",

                "PMC Demand": pmc,
                "UR": ur_rate,
                "Prod Mult (1-UR)": 1.0 - ur_rate,
                "Production Demand": pd_prod,

                "Group Demand": gr,
                "Group Count": gc,
                "People per Group": ppl_per_group,

                "Actual": actual,
                "Turnover Rate": to_rate,
                "Absent": ab_rate,
                "HC net": net_after,                         # hiển thị HC net = NetAfterTransfer
                "Hiring": hire,

                "Gap": gap_after,
                "HC net before hiring": hc_net_before_hiring_after,

                "In": infl_cell,
                "Out": outf_cell,

                "Slack HC": shc,

                "HC net (no transfer)": net_pre,
                "Gap (no transfer)": gap_pre,
                "HC net before hiring (no transfer)": hc_net_before_hiring_pre,
            })

    df_bucket = pd.DataFrame(rows).sort_values(["Week", "Zone", "Family"]).reset_index(drop=True)

    # Transfers long (giữ như bạn đang làm)
    x_rows = []
    for t in T:
        for e in m.E:
            val_raw = float(pyo.value(m.X[t, e]))
            val_int = int(round(val_raw))  # report only

            if val_int == 0:
                continue  # không ghi lên output nếu làm tròn ra 0

            zi, fi, zj, fj = e
            score = int(pyo.value(m.ScoreE[e])) if hasattr(m, "ScoreE") else 1

            x_rows.append({
                "Week": int(t),
                "From GF": f"{zi} | {fi}",
                "To GF": f"{zj} | {fj}",
                "From Zone": str(zi),
                "To Zone": str(zj),
                "Zone Transfer Score": score,  # NEW
                "Transfer HC": val_int,
                "Transfer Score (HC*Score)": val_int * score,  # NEW (rõ hơn)
            })

    df_x_long = pd.DataFrame(
        x_rows,
        columns=[
            "Week", "From GF", "To GF",
            "From Zone", "To Zone",
            "Zone Transfer Score",
            "Transfer HC",
            "Transfer Score (HC*Score)",
        ]
    )

    # KPI đơn giản
    kpi = pd.DataFrame([{
        "total_production_demand": float(df_bucket["Production Demand"].sum()),
        "total_hiring": float(df_bucket["Hiring"].sum()),
        "total_transfer": float(df_x_long["Transfer HC"].sum()) if not df_x_long.empty else 0.0,
        "total_transfer_score": float(df_x_long["Transfer Score (HC*Score)"].sum()) if not df_x_long.empty else 0.0,
        "total_slack_hc": float(df_bucket["Slack HC"].sum()),
    }])

    return {
        "KPI": kpi,
        "Bucket Weekly (Long)": df_bucket,
        "Transfers Long": df_x_long,
    }



def _style_excel(path: str) -> None:
    # Optional: make output prettier (bold header, freeze panes, autofilter, width)
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except Exception:
        return

    wb = load_workbook(path)

    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="D9E1F2")  # light blue
    hire_fill = PatternFill("solid", fgColor="92D050")  # xanh
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for ws in wb.worksheets:
        if ws.max_row >= 1:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # style header row
            for c in range(1, ws.max_column + 1):
                cell = ws.cell(row=1, column=c)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_align
            from openpyxl.styles import PatternFill
            from openpyxl.utils import get_column_letter
            from openpyxl.formatting.rule import CellIsRule

            hire_fill = PatternFill("solid", fgColor="92D050")

            def _apply_hire_conditional(ws, row_idx: int):
                # giả định cột A là label, từ cột B trở đi là tuần / số liệu
                start_col = 2
                end_col = ws.max_column
                if end_col < start_col:
                    return

                # XÓA fill tĩnh (nếu trước đó bạn đã tô cả hàng)
                for c in range(start_col, end_col + 1):
                    ws.cell(row=row_idx, column=c).fill = PatternFill()

                rng = f"{get_column_letter(start_col)}{row_idx}:{get_column_letter(end_col)}{row_idx}"
                rule = CellIsRule(operator="greaterThan", formula=["0"], fill=hire_fill)
                ws.conditional_formatting.add(rng, rule)

            # ---- áp dụng cho các dòng có label Hiring / Total Hiring
            for r in range(1, ws.max_row + 1):
                v = ws.cell(row=r, column=1).value
                if v is None:
                    continue
                label = str(v).strip().lower()
                if label in {"hiring", "total hiring"}:
                    _apply_hire_conditional(ws, r)

                    # (tuỳ chọn) giữ ô label (cột A) luôn xanh để nhìn ra hàng Hiring
                    ws.cell(row=r, column=1).fill = hire_fill
                    ws.cell(row=r, column=1).font = Font(bold=True)

            # column widths
            for c in range(1, ws.max_column + 1):
                letter = get_column_letter(c)
                max_len = 0
                for r in range(1, min(ws.max_row, 200) + 1):  # avoid super slow on huge sheets
                    v = ws.cell(row=r, column=c).value
                    if v is None:
                        continue
                    max_len = max(max_len, len(str(v)))
                ws.column_dimensions[letter].width = min(45, max(10, max_len + 2))

    wb.save(path)

def _add_gf_dashboard_sheet(wb, df_bucket: pd.DataFrame, df_transfers: Optional[pd.DataFrame] = None) -> None:
    from copy import copy
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    name = "GF Dashboard"
    if name in wb.sheetnames:
        wb.remove(wb[name])
    ws = wb.create_sheet(name, 0)

    # ---- base sets
    weeks = sorted(df_bucket["Week"].unique().tolist())
    gfs = sorted(df_bucket["GF"].drop_duplicates().tolist())

    # ---- Styles
    thin = Side(style="thin", color="000000")
    border_thin = Border(left=thin, right=thin, top=thin, bottom=thin)

    thick = Side(style="medium", color="000000")  # đổi "thick" nếu muốn đậm hơn

    def _outline_range(r1: int, c1: int, r2: int, c2: int) -> None:
        """Vẽ viền đậm rìa ngoài cho vùng (r1..r2, c1..c2) và giữ viền trong."""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = ws.cell(r, c)
                b = cell.border if cell.border else Border()
                nb = copy(b)

                if c == c1:
                    nb.left = thick
                if c == c2:
                    nb.right = thick
                if r == r1:
                    nb.top = thick
                if r == r2:
                    nb.bottom = thick

                cell.border = nb

    fill_header = PatternFill("solid", fgColor="7F7F7F")   # dark grey
    fill_demand = PatternFill("solid", fgColor="F4B183")   # orange
    fill_actual = PatternFill("solid", fgColor="9DC3E6")   # light blue
    fill_hire   = PatternFill("solid", fgColor="92D050")   # green
    fill_gray   = PatternFill("solid", fgColor="D9D9D9")   # grey

    # đỏ nhạt đúng #FF9999 => ARGB = FFFF9999
    fill_neg = PatternFill("solid", fgColor="FFFFCCCC")
    f_neg = Font(name="Aptos Narrow", bold=True, color="9C0006")  # dark red

    FONT_NAME = "Aptos Narrow"
    f_header = Font(name=FONT_NAME, bold=True, color="FFFFFF")
    f_title  = Font(name=FONT_NAME, bold=True, size=12)
    f_bold   = Font(name=FONT_NAME, bold=True)
    f_italic_gray = Font(name=FONT_NAME, italic=True, color="7F7F7F")

    align_center = Alignment(horizontal="center", vertical="center")
    align_left   = Alignment(horizontal="left", vertical="center")

    # Column widths
    ws.column_dimensions["A"].width = 35
    for j in range(2, 2 + len(weeks)):
        ws.column_dimensions[get_column_letter(j)].width = 12

    end_col = 1 + len(weeks)  # A..(B..)

    # ---- helpers
    def apply_neg_if_negative(row_idx: int):
        """Tô thủ công: nếu ô <0 => nền đỏ nhạt + font đỏ. Đồng thời set format hiển thị số âm."""
        for j in range(2, 2 + len(weeks)):
            c = ws.cell(row_idx, j)

            c.number_format = "0;[Red]-0;0"

            if c.value is None or c.value == "":
                continue
            try:
                v = float(c.value)
            except Exception:
                continue

            if v < 0:
                c.fill = fill_neg
                c.font = f_neg

    def write_row_int(
        r: int,
        label: str,
        values: Dict[int, float],
        fill: Optional[PatternFill] = None,
        bold: bool = False,
        blank_zero: bool = False,
        label_font: Optional[Font] = None,
    ):
        # label cell
        lc = ws.cell(r, 1, label)
        lc.alignment = align_left
        lc.border = border_thin
        lc.font = label_font if label_font is not None else (f_bold if bold else Font(name=FONT_NAME))
        if fill:
            lc.fill = fill

        # week cells
        for j, w in enumerate(weeks, start=2):
            v = values.get(w, 0.0)
            v = 0.0 if v is None else float(v)
            v = int(round(v))
            out_v = "" if (blank_zero and v == 0) else v

            c = ws.cell(r, j, out_v)
            c.number_format = "0"
            c.alignment = align_center
            c.border = border_thin
            if fill:
                c.fill = fill
            c.font = f_bold if bold else Font(name=FONT_NAME)

    def write_row_pct(
        r: int,
        label: str,
        values: Dict[int, float],
        fill: Optional[PatternFill] = None,
        bold: bool = False,
    ):
        lc = ws.cell(r, 1, label)
        lc.alignment = align_left
        lc.border = border_thin
        lc.font = f_bold if bold else Font(name=FONT_NAME)
        if fill:
            lc.fill = fill

        for j, w in enumerate(weeks, start=2):
            v = values.get(w, 0.0)
            v = 0.0 if v is None else float(v)
            c = ws.cell(r, j, v)
            c.number_format = "0%"
            c.alignment = align_center
            c.border = border_thin
            if fill:
                c.fill = fill
            c.font = f_bold if bold else Font(name=FONT_NAME)

    # ---- transfer lookup
    df_tr = df_transfers if (df_transfers is not None and not df_transfers.empty) else None

    def out_to_map(from_gf: str, to_gf: str) -> Dict[int, float]:
        if df_tr is None:
            return {w: 0.0 for w in weeks}
        sub = df_tr[(df_tr["From GF"] == from_gf) & (df_tr["To GF"] == to_gf)]
        mp = sub.groupby("Week")["Transfer HC"].sum().to_dict()
        return {w: float(mp.get(w, 0.0)) for w in weeks}

    def out_total_map(from_gf: str) -> Dict[int, float]:
        if df_tr is None:
            return {w: 0.0 for w in weeks}
        sub = df_tr[df_tr["From GF"] == from_gf]
        mp = sub.groupby("Week")["Transfer HC"].sum().to_dict()
        return {w: float(mp.get(w, 0.0)) for w in weeks}

    def in_from_map(to_gf: str, from_gf: str) -> Dict[int, float]:
        if df_tr is None:
            return {w: 0.0 for w in weeks}
        sub = df_tr[(df_tr["To GF"] == to_gf) & (df_tr["From GF"] == from_gf)]
        mp = sub.groupby("Week")["Transfer HC"].sum().to_dict()
        return {w: float(mp.get(w, 0.0)) for w in weeks}

    def in_total_map(to_gf: str) -> Dict[int, float]:
        if df_tr is None:
            return {w: 0.0 for w in weeks}
        sub = df_tr[df_tr["To GF"] == to_gf]
        mp = sub.groupby("Week")["Transfer HC"].sum().to_dict()
        return {w: float(mp.get(w, 0.0)) for w in weeks}

    # ---- Build sheet
    row = 1
    for gf in gfs:
        sub = df_bucket[df_bucket["GF"] == gf].set_index("Week")

        # Title
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=end_col)
        tcell = ws.cell(row, 1, gf)
        tcell.font = f_title
        tcell.alignment = align_left
        row += 1

        # Week header row (outline đậm)
        week_row = row
        ws.cell(row, 1, "Week").fill = fill_header
        ws.cell(row, 1).font = f_header
        ws.cell(row, 1).alignment = align_center
        ws.cell(row, 1).border = border_thin

        for j, w in enumerate(weeks, start=2):
            c = ws.cell(row, j, w)
            c.fill = fill_header
            c.font = f_header
            c.alignment = align_center
            c.border = border_thin

        _outline_range(week_row, 1, week_row, end_col)
        row += 1

        # Demand block: PMC -> Production Demand (outline đậm)
        demand_start = row
        write_row_int(row, "PMC demand", sub["PMC Demand"].to_dict(), fill=fill_demand, bold=True); row += 1
        write_row_pct(row, "%Under Routing", sub["UR"].to_dict(), fill=fill_demand, bold=True); row += 1
        write_row_int(row, "Production Demand", sub["Production Demand"].to_dict(), fill=fill_demand, bold=True); row += 1
        demand_end = row - 1
        _outline_range(demand_start, 1, demand_end, end_col)

        # Given block: Given -> details (outline đậm)
        given_start = row
        write_row_int(row, "Given", out_total_map(gf), fill=fill_demand, bold=True); row += 1
        for to_gf in [x for x in gfs if x != gf]:
            write_row_int(
                row, to_gf, out_to_map(gf, to_gf),
                fill=None, bold=False, blank_zero=True,
                label_font=f_italic_gray
            )
            row += 1
        given_end = row - 1
        _outline_range(given_start, 1, given_end, end_col)

        # Actual block: Actual -> HC net (outline đậm)
        actual_start = row
        write_row_int(row, "Actual", sub["Actual"].to_dict(), fill=fill_actual, bold=True); row += 1
        write_row_pct(row, "Turn over rate", sub["Turnover Rate"].to_dict(), fill=fill_actual, bold=True); row += 1
        write_row_pct(row, "Absent", sub["Absent"].to_dict(), fill=fill_actual, bold=True); row += 1
        write_row_int(row, "HC net", sub["HC net (no transfer)"].to_dict(), fill=fill_actual, bold=True); row += 1
        actual_end = row - 1
        _outline_range(actual_start, 1, actual_end, end_col)

        # Hiring + Borrow block: Hiring -> Borrow details (outline đậm)
        hb_start = row
        write_row_int(row, "Hiring", sub["Hiring"].to_dict(), fill=fill_hire, bold=True)
        ws.cell(row, 1).fill = fill_hire
        row += 1

        write_row_int(row, "Borrow", in_total_map(gf), fill=fill_hire, bold=True); row += 1
        for from_gf in [x for x in gfs if x != gf]:
            write_row_int(
                row, from_gf, in_from_map(gf, from_gf),
                fill=None, bold=False, blank_zero=True,
                label_font=f_italic_gray
            )
            row += 1
        hb_end = row - 1
        _outline_range(hb_start, 1, hb_end, end_col)

        # Gap block: Gap before -> HC net before hiring (outline đậm + tô đỏ âm)
        gap_start = row
        write_row_int(row, "Gap before transfer", sub["Gap (no transfer)"].to_dict(), fill=fill_gray, bold=True)
        apply_neg_if_negative(row)
        row += 1

        write_row_int(row, "Gap after transfer", sub["Gap"].to_dict(), fill=fill_gray, bold=True)
        apply_neg_if_negative(row)
        row += 1

        write_row_int(row, "HC net before hiring", sub["HC net before hiring (no transfer)"].to_dict(), fill=fill_gray, bold=True)
        row += 1

        gap_end = row - 1
        _outline_range(gap_start, 1, gap_end, end_col)

        row += 2  # spacing between GFs

    # Summary at bottom (giữ như cũ, không yêu cầu viền đậm)
    sum_df = df_bucket.groupby("Week", as_index=True).agg({
        "Production Demand": "sum",
        "Actual": "sum",
        "Hiring": "sum",
        "HC net (no transfer)": "sum",
        "Gap (no transfer)": "sum",
    })

    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=end_col)
    scell = ws.cell(row, 1, "TOTAL (All GFs)")
    scell.font = f_title
    scell.alignment = align_left
    row += 1

    # Week header again
    ws.cell(row, 1, "Week").fill = fill_header
    ws.cell(row, 1).font = f_header
    ws.cell(row, 1).alignment = align_center
    ws.cell(row, 1).border = border_thin
    for j, w in enumerate(weeks, start=2):
        c = ws.cell(row, j, w)
        c.fill = fill_header
        c.font = f_header
        c.alignment = align_center
        c.border = border_thin
    row += 1

    write_row_int(row, "Total Production Demand", sum_df["Production Demand"].to_dict(), fill=fill_demand, bold=True); row += 1
    write_row_int(row, "Total Actual", sum_df["Actual"].to_dict(), fill=fill_actual, bold=True); row += 1
    write_row_int(row, "Total HC net", sum_df["HC net (no transfer)"].to_dict(), fill=fill_actual, bold=True); row += 1
    write_row_int(row, "Total Hiring", sum_df["Hiring"].to_dict(), fill=fill_hire, bold=True)
    ws.cell(row, 1).fill = fill_hire
    row += 1

    write_row_int(row, "Total Gap", sum_df["Gap (no transfer)"].to_dict(), fill=fill_gray, bold=True)

def _add_transfer_matrix_sheet(wb, df_x_long: pd.DataFrame) -> None:
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    name = "Transfer Matrix"
    if name in wb.sheetnames:
        wb.remove(wb[name])
    ws = wb.create_sheet(name)

    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    fill_header = PatternFill("solid", fgColor="7F7F7F")
    f_header = Font(bold=True, color="FFFFFF")
    f_title = Font(bold=True, size=12)
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")

    if df_x_long.empty:
        ws["A1"] = "No transfers"
        return

    weeks = sorted(df_x_long["Week"].unique().tolist())
    gfs = sorted(set(df_x_long["From GF"]).union(set(df_x_long["To GF"])))

    row = 1
    for w in weeks:
        block = df_x_long[df_x_long["Week"] == w]

        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2 + len(gfs))
        c = ws.cell(row, 1, f"Week {w} - Transfer In/Out Matrix")
        c.font = f_title
        c.alignment = align_left
        row += 1

        # header row
        h = ws.cell(row, 1, "From \\ To")
        h.fill = fill_header
        h.font = f_header
        h.alignment = align_center
        h.border = border

        for j, to_gf in enumerate(gfs, start=2):
            cc = ws.cell(row, j, to_gf)
            cc.fill = fill_header
            cc.font = f_header
            cc.alignment = align_center
            cc.border = border
        row += 1

        # build lookup
        pivot = block.pivot_table(index="From GF", columns="To GF", values="Transfer HC", aggfunc="sum", fill_value=0.0)

        for i, from_gf in enumerate(gfs, start=0):
            rr = row + i
            lc = ws.cell(rr, 1, from_gf)
            lc.alignment = align_left
            lc.border = border

            for j, to_gf in enumerate(gfs, start=2):
                val = float(pivot.loc[from_gf, to_gf]) if (from_gf in pivot.index and to_gf in pivot.columns) else 0.0
                cc = ws.cell(rr, j, val if val != 0 else "")
                cc.number_format = "0.00"
                cc.alignment = align_center
                cc.border = border

        row += len(gfs) + 2  # space between weeks

    ws.column_dimensions["A"].width = 28
    # các cột còn lại auto-ish
    from openpyxl.utils import get_column_letter
    for col in range(2, 2 + len(gfs)):
        ws.column_dimensions[get_column_letter(col)].width = 18

def write_output_excel(frames: Dict[str, pd.DataFrame], output_path: str) -> None:
    # 1) write raw sheets by pandas
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    # 2) post-process workbook: add formatted sheets
    from openpyxl import load_workbook
    wb = load_workbook(output_path)

    df_bucket = frames.get("Bucket Weekly (Long)")
    df_transfers = frames.get("Transfers Long")

    if df_bucket is not None and not df_bucket.empty:
        _add_gf_dashboard_sheet(wb, df_bucket, df_transfers)

    if df_transfers is not None:
        _add_transfer_matrix_sheet(wb, df_transfers)

    wb.save(output_path)

def run_from_excel(input_path: str, output_path: str) -> Dict[str, pd.DataFrame]:
    data = read_input_excel(input_path)
    model = build_model(data)
    res = solve_model(model, tie_break_transfers=data.settings.tie_break_transfers, epsilon=data.settings.epsilon)

    frames = extract_results(data, model)
    frames["Solve Info"] = pd.DataFrame([{
        "status": res.status,
        "termination_condition": res.termination_condition,
        "total_slack_hc": res.total_shortage,
        "total_hiring": res.total_hiring,
        "total_transfer": res.total_transfer,
        "big_m": float(pyo.value(model.BigM)),
        "num_edges": len(list(model.E)),
        "total_transfer_score": res.total_transfer_score,
    }])

    write_output_excel(frames, output_path)
    return frames


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Path to input Excel workbook")
    parser.add_argument("--output", required=False, help="Path to output Excel workbook")
    parser.add_argument("--timestamp_output", action="store_true", help="Add timestamp to output filename (avoid Excel lock)")
    args = parser.parse_args()

    default_input = os.path.join(os.getcwd(), "input_template_v5.xlsx")
    default_output = os.path.join(os.getcwd(), "output5.xlsx")

    input_path = args.input or default_input

    if args.output:
        output_path = args.output
    else:
        if args.timestamp_output:
            base, ext = os.path.splitext(default_output)
            output_path = f"{base}_{int(time.time())}{ext}"
        else:
            output_path = default_output

    print("Running with:")
    print("  input :", input_path)
    print("  output:", output_path)

    run_from_excel(input_path, output_path)
    print(f"Done. Wrote: {output_path}")
