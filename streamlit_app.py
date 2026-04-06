from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
import io

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FAO Diagnostic Test", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def img_to_base64(path: str) -> str:
    file_path = Path(__file__).parent / path
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_b64 = img_to_base64("EAB-Logo-RGB.png")


# ----------------------------
# Header + Global CSS
# ----------------------------
st.markdown(
    f"""
    <style>
      header[data-testid="stHeader"] {{ display: none; }}
      [data-testid="stToolbar"] {{ display: none; }}
      [data-testid="stDecoration"] {{ display: none; }}
      [data-testid="stMainBlockContainer"] {{
        padding-top: 1.0rem !important;
        margin-top: 0 !important;
      }}

      .brand-wrap {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 6px 0 10px 0;
      }}
      .brand-logo {{
        height: 56px;
        width: auto;
        display: block;
      }}
      .brand-divider {{
        width: 1px;
        height: 46px;
        background: #CBD5E1;
        margin: 0 2px;
      }}
      .brand-text {{
        display: flex;
        flex-direction: column;
        line-height: 1.05;
      }}
      .brand-title {{
        font-size: 32px;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #2F3A45;
        margin: 0;
        padding: 0;
      }}
      .brand-subtitle {{
        font-size: 16px;
        font-weight: 500;
        color: #9AA3AE;
        margin-top: 6px;
      }}
    </style>

    <div class="brand-wrap">
      <img class="brand-logo" src="data:image/png;base64,{logo_b64}" />
      <div class="brand-divider"></div>
      <div class="brand-text">
        <div class="brand-title">FAO Diagnostic Test</div>
        <div class="brand-subtitle">Data Validation Platform</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Constants (lowercase)
# -----------------------------
REQUIRED_FIELDS_BASE = [
    "studentid",
    "efcrank",
    "matric",
    "resident",
    "hdefc",
    "hdneed",
    "pellft",
    "stateft",
    "totpts",
    "instmeritft",
    "tgrant",
]


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ValidationConfig:
    stratification_vars: Tuple[str, ...]
    best_efc: str
    best_need: str
    best_academic: str
    positive_iar_correlation: str
    best_pell: str
    best_state: str
    best_merit: str
    partner_name: str


# -----------------------------
# Helpers
# -----------------------------
def N(x: str) -> str:
    return str(x).strip().lower()


def is_no(x: str) -> bool:
    return str(x).strip().lower() == "no"


def parse_strat_vars(text: str) -> Tuple[str, ...]:
    if not text:
        return tuple()
    t = str(text).strip()
    if t.lower() in ("none", "no"):
        return tuple()
    return tuple(N(v) for v in t.split(",") if v.strip())


def sheet_name_ci(xls: pd.ExcelFile, desired: str) -> Optional[str]:
    desired_lower = desired.strip().lower()
    for sheet in xls.sheet_names:
        if sheet.strip().lower() == desired_lower:
            return sheet
    return None


def normalize_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [N(c) for c in df.columns]
    return df


def coerce_studentid(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns_lower(df)
    candidates = ["studentid", "student_id", "student id"]
    found = next((c for c in candidates if c in df.columns), None)

    if found is None:
        raise ValueError(
            "Missing student id column (expected STUDENTID / STUDENT_ID / STUDENT ID)."
        )

    if found != "studentid":
        df.rename(columns={found: "studentid"}, inplace=True)

    return df


@st.cache_data(show_spinner=False)
def read_upload_cached(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    fname = file_name.lower().strip()

    if fname.endswith(".csv"):
        df = pd.read_csv(buf)
        return coerce_studentid(df)

    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        xls = pd.ExcelFile(buf, engine="calamine")
        data_sheet = sheet_name_ci(xls, "Data")
        amp_sheet = sheet_name_ci(xls, "AMPData")

        if data_sheet is None and amp_sheet is None:
            raise ValueError(
                'Excel must contain at least one sheet named "Data" or "AMPData" '
                "(case-insensitive)."
            )

        df_data = (
            coerce_studentid(pd.read_excel(xls, sheet_name=data_sheet))
            if data_sheet
            else None
        )
        df_amp = (
            coerce_studentid(pd.read_excel(xls, sheet_name=amp_sheet))
            if amp_sheet
            else None
        )

        if df_data is None:
            return df_amp
        if df_amp is None:
            return df_data

        merged = df_data.merge(df_amp, on="studentid", how="outer", suffixes=("", "_amp"))
        return normalize_columns_lower(merged)

    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def validate_required_fields(df: pd.DataFrame, required_cols: List[str]) -> None:
    cols = {N(c) for c in df.columns}
    missing = [c for c in required_cols if N(c) not in cols]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        c = N(c)
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def keep_only(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    keep_cols = [N(c) for c in keep_cols]
    existing = [c for c in keep_cols if c in df.columns]
    return df.loc[:, existing].copy()


def build_summary(
    df: pd.DataFrame,
    group_cols: List[str],
    mean_cols: List[str],
    include_efcrank_1: bool,
) -> pd.DataFrame:
    dfx = df.copy()
    if not include_efcrank_1:
        dfx = dfx.loc[dfx["efcrank"] != 1].copy()

    return (
        dfx.groupby(group_cols, dropna=False)
        .agg(**{f"mean_{c}": (c, "mean") for c in mean_cols}, n=("studentid", "count"))
        .reset_index()
    )


def pair_diffs(
    summary: pd.DataFrame,
    group_without_matric: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    s = summary.copy()
    s["matric"] = pd.to_numeric(s["matric"], errors="coerce")

    s0 = s[s["matric"] == 0].copy()
    s1 = s[s["matric"] == 1].copy()

    s0 = s0[group_without_matric + [f"mean_{m}" for m in metric_cols] + ["n"]].rename(
        columns={**{f"mean_{m}": f"mean0_{m}" for m in metric_cols}, "n": "n0"}
    )
    s1 = s1[group_without_matric + [f"mean_{m}" for m in metric_cols] + ["n"]].rename(
        columns={**{f"mean_{m}": f"mean1_{m}" for m in metric_cols}, "n": "n1"}
    )

    merged = pd.merge(s0, s1, on=group_without_matric, how="outer")
    merged["n0"] = merged["n0"].fillna(0).astype(int)
    merged["n1"] = merged["n1"].fillna(0).astype(int)

    for m in metric_cols:
        merged[f"diff_{m}"] = merged[f"mean0_{m}"] - merged[f"mean1_{m}"]

    return merged


# -----------------------------
# Severity logic
# -----------------------------
def severity_state_or_pell_core(
    row: pd.Series,
    diff_need_col: str,
    diff_fund_col: str,
) -> Tuple[Optional[str], Optional[str]]:
    if row.get("n0", 0) < 15 or row.get("n1", 0) < 15:
        return None, None

    d_need = row.get(diff_need_col)
    d_fund = row.get(diff_fund_col)
    max_need = row.get("max_need")

    if pd.isna(d_need) or pd.isna(d_fund) or pd.isna(max_need):
        return None, None

    if max_need == 1 and d_fund <= -200:
        return "MAJOR", "MAJOR_MAXNEED"
    if d_need < 0 and d_fund < d_need:
        return "MAJOR", "MAJOR_NEED_NEG"
    if max_need == 0 and d_need > 0 and d_fund <= -200:
        return "MAJOR", "MAJOR_NEED_POS"
    if d_need >= -200 and d_fund <= -300:
        return "INTERMEDIATE", "INTERMEDIATE"
    if d_need >= -100 and d_fund <= -150:
        return "MINOR", "MINOR"

    return None, None


def severity_merit(
    row: pd.Series,
    positive_corr_yes: bool,
    acad: str,
    merit: str,
) -> Tuple[Optional[str], Optional[str]]:
    if row.get("n0", 0) < 15 or row.get("n1", 0) < 15:
        return None, None

    d_acad = row.get(f"diff_{acad}")
    d_merit = row.get(f"diff_{merit}")
    m0 = row.get(f"mean0_{acad}")
    m1 = row.get(f"mean1_{acad}")

    if pd.isna(d_acad) or pd.isna(d_merit) or pd.isna(m0) or pd.isna(m1) or m1 == 0:
        return None, None

    ratio = m0 / m1

    if not positive_corr_yes:
        if d_acad < 0 and d_merit <= -200:
            return "MAJOR", "MAJOR"
        if ratio >= 1.02 and d_merit <= -300:
            return "INTERMEDIATE", "INTERMEDIATE"
        if ratio >= 1.02 and d_merit <= -150:
            return "MINOR", "MINOR"
    else:
        if d_acad > 0 and d_merit <= -200:
            return "MAJOR", "MAJOR"
        if ratio >= 0.98 and d_merit <= -300:
            return "INTERMEDIATE", "INTERMEDIATE"
        if ratio >= 0.98 and d_merit <= -150:
            return "MINOR", "MINOR"

    return None, None


def state_note(code: str) -> str:
    return {
        "MAJOR_MAXNEED": "These students are Aid Applicants whose EFC ≤ 0. I would expect the difference in the mean State funding values to be smaller. The State funding field may be incompletely populated.",
        "MAJOR_NEED_NEG": "The difference in State funding exceeds the observed difference in financial need. The State funding field may be incompletely populated.",
        "MAJOR_NEED_POS": "The MATRIC=0 admits receive less State funding despite demonstrating greater financial need. The State funding field may be incompletely populated.",
        "INTERMEDIATE": "These students are Aid Applicants where MATRIC=0 and who have essentially the same level of need, yet they receive less State funding than do their MATRIC=1 counterparts. The State funding field may be incompletely populated.",
        "MINOR": "These students are Aid Applicants where MATRIC=0 and who have greater levels of financial need, yet they receive somewhat less State funding than do their MATRIC=1 counterparts. The State funding field may be incompletely populated.",
    }[code]


def pell_note(code: str) -> str:
    return {
        "MAJOR_MAXNEED": "These students are Aid Applicants whose EFC ≤ 0. I would expect the difference in the mean Pell Grant values to be smaller. The Pell Grant field may be incompletely populated.",
        "MAJOR_NEED_NEG": "The difference in Pell Grant funding exceeds the observed difference in financial need. The Pell Grant field may be incompletely populated.",
        "MAJOR_NEED_POS": "The MATRIC=0 admits receive less Pell Grant funding despite demonstrating greater financial need. The Pell Grant field may be incompletely populated.",
        "INTERMEDIATE": "These students are Aid Applicants where MATRIC=0 and who have essentially the same level of financial need, yet they receive less Pell Grant funding than do their MATRIC=1 counterparts. The Pell Grant field may be incompletely populated.",
        "MINOR": "These students are Aid Applicants where MATRIC=0 and who have greater levels of financial need, yet they receive somewhat less Pell Grant funding than do their MATRIC=1 counterparts. The Pell Grant field may be incompletely populated.",
    }[code]


def merit_note(sev: str) -> str:
    if sev == "MAJOR":
        return "The MATRIC = 0 admits demonstrate higher academic achievement yet receive less in Institutional Aid than do their MATRIC=1 counterparts. The Institutional Merit field may not be populated completely."
    if sev == "INTERMEDIATE":
        return "The MATRIC = 0 admits demonstrate essentially equivalent academic achievement yet receive less in Institutional Aid than do their MATRIC=1 counterparts. The Institutional Merit field may not be populated completely."
    if sev == "MINOR":
        return "The MATRIC = 0 admits demonstrate essentially equivalent academic achievement yet receive somewhat less in Institutional Aid than do their MATRIC=1 counterparts. The Institutional Merit field may not be populated completely."
    raise KeyError(sev)


@st.cache_data(show_spinner=False)
def run_validation_cached(df_raw: pd.DataFrame, cfg_dict: Dict) -> Dict[str, pd.DataFrame]:
    cfg = ValidationConfig(**cfg_dict)
    df_raw = normalize_columns_lower(df_raw)

    strat_vars = [N(v) for v in cfg.stratification_vars]
    best_efc = N(cfg.best_efc)
    best_need = N(cfg.best_need)
    best_acad = N(cfg.best_academic)
    best_pell = N(cfg.best_pell)
    best_state = N(cfg.best_state)
    best_merit = N(cfg.best_merit)
    pos_corr_yes = str(cfg.positive_iar_correlation).strip().title() == "Yes"

    required = sorted(
        set(
            REQUIRED_FIELDS_BASE
            + strat_vars
            + [best_efc, best_need, best_acad, best_pell, best_state, best_merit]
        )
    )
    validate_required_fields(df_raw, required)

    df = keep_only(df_raw, required)
    df = to_numeric(
        df,
        [
            "efcrank",
            "matric",
            "resident",
            best_efc,
            best_need,
            best_acad,
            best_pell,
            best_state,
            best_merit,
        ],
    )

    for c in [best_pell, best_state, best_efc, best_merit]:
        df[c] = df[c].fillna(0)

    df["max_need"] = ((df["efcrank"] >= 2) & (df[best_efc] <= 0)).astype(int)

    state_group = strat_vars + ["resident", "efcrank", "max_need", "matric"]
    state_summary = build_summary(df, state_group, [best_need, best_state], include_efcrank_1=False)
    state_pairs = pair_diffs(
        state_summary,
        strat_vars + ["resident", "efcrank", "max_need"],
        [best_need, best_state],
    )

    def state_severity_with_resident_gate(r: pd.Series) -> Tuple[Optional[str], Optional[str]]:
        res = r.get("resident")
        if pd.isna(res) or int(res) != 1:
            return None, None
        return severity_state_or_pell_core(r, f"diff_{best_need}", f"diff_{best_state}")

    state_pairs[["severity", "message_code"]] = pd.DataFrame(
        state_pairs.apply(state_severity_with_resident_gate, axis=1).tolist(),
        index=state_pairs.index,
    )
    state_anoms = state_pairs[state_pairs["severity"].notna()].copy()
    if not state_anoms.empty:
        state_anoms["note"] = state_anoms["message_code"].map(state_note)

    pell_group = strat_vars + ["efcrank", "max_need", "matric"]
    pell_summary = build_summary(df, pell_group, [best_need, best_pell], include_efcrank_1=False)
    pell_pairs = pair_diffs(
        pell_summary,
        strat_vars + ["efcrank", "max_need"],
        [best_need, best_pell],
    )
    pell_pairs[["severity", "message_code"]] = pd.DataFrame(
        pell_pairs.apply(
            lambda r: severity_state_or_pell_core(r, f"diff_{best_need}", f"diff_{best_pell}"),
            axis=1,
        ).tolist(),
        index=pell_pairs.index,
    )
    pell_anoms = pell_pairs[pell_pairs["severity"].notna()].copy()
    if not pell_anoms.empty:
        pell_anoms["note"] = pell_anoms["message_code"].map(pell_note)

    merit_group = strat_vars + ["efcrank", "matric"]
    merit_summary = build_summary(df, merit_group, [best_acad, best_merit], include_efcrank_1=True)
    merit_pairs = pair_diffs(merit_summary, strat_vars + ["efcrank"], [best_acad, best_merit])
    merit_pairs[["severity", "message_code"]] = pd.DataFrame(
        merit_pairs.apply(
            lambda r: severity_merit(r, pos_corr_yes, best_acad, best_merit),
            axis=1,
        ).tolist(),
        index=merit_pairs.index,
    )
    merit_anoms = merit_pairs[merit_pairs["severity"].notna()].copy()
    if not merit_anoms.empty:
        merit_anoms["note"] = merit_anoms["severity"].map(merit_note)

    return {
        "clean_data": df,
        "state_summary": state_summary,
        "state_anomalies": state_anoms,
        "pell_summary": pell_summary,
        "pell_anomalies": pell_anoms,
        "merit_summary": merit_summary,
        "merit_anomalies": merit_anoms,
    }


def build_output_sheets(results: Dict[str, pd.DataFrame], cfg_dict: Dict) -> Dict[str, pd.DataFrame]:
    cfg = ValidationConfig(**cfg_dict)
    strat_vars = [N(v) for v in cfg.stratification_vars]

    best_need = N(cfg.best_need)
    best_pell = N(cfg.best_pell)
    best_state = N(cfg.best_state)
    best_acad = N(cfg.best_academic)
    best_merit = N(cfg.best_merit)

    def attach(summary: pd.DataFrame, anoms: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
        out = summary.copy()
        if anoms is None or anoms.empty or "note" not in anoms.columns:
            out["severity"] = ""
            out["note"] = ""
            return out

        an = anoms[key_cols + ["severity", "note"]].copy()
        out = out.merge(an, on=key_cols, how="left")
        out["severity"] = out["severity"].fillna("")
        out["note"] = out["note"].fillna("")
        return out

    state = results["state_summary"].copy()
    state = state.rename(
        columns={f"mean_{best_need}": "financial_need", f"mean_{best_state}": "state_funding"}
    )
    state = attach(state, results["state_anomalies"], strat_vars + ["resident", "efcrank", "max_need"])

    pell = results["pell_summary"].copy()
    pell = pell.rename(
        columns={f"mean_{best_need}": "financial_need", f"mean_{best_pell}": "pell_grant"}
    )
    pell = attach(pell, results["pell_anomalies"], strat_vars + ["efcrank", "max_need"])

    merit = results["merit_summary"].copy()
    merit = merit.rename(
        columns={
            f"mean_{best_acad}": "academic_achievement",
            f"mean_{best_merit}": "institutional_merit",
        }
    )
    merit = attach(merit, results["merit_anomalies"], strat_vars + ["efcrank"])

    if "academic_achievement" in merit.columns:
        merit["academic_achievement"] = pd.to_numeric(
            merit["academic_achievement"], errors="coerce"
        ).round(1)

    return {"State": state, "Pell": pell, "Merit": merit}


def sheets_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet_name = name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            headers = [cell.value for cell in ws[1]]
            col_idx = {
                h: i + 1 for i, h in enumerate(headers) if isinstance(h, str) and h.strip()
            }

            fmt_int_comma = "#,##0"
            fmt_1dp = "0.0"

            comma_cols = {
                "financial_need",
                "state_funding",
                "pell_grant",
                "institutional_merit",
                "n",
                "n0",
                "n1",
            }

            for h in headers:
                if not isinstance(h, str):
                    continue
                hl = h.lower()
                if hl.startswith("diff_") or hl.startswith("mean0_") or hl.startswith("mean1_"):
                    comma_cols.add(h)

            one_decimal_cols = set()
            if sheet_name.lower() == "merit" and "academic_achievement" in col_idx:
                one_decimal_cols.add("academic_achievement")

            max_row = ws.max_row

            for col_name in comma_cols:
                if col_name in col_idx:
                    c = col_idx[col_name]
                    for r in range(2, max_row + 1):
                        ws.cell(row=r, column=c).number_format = fmt_int_comma

            for col_name in one_decimal_cols:
                c = col_idx[col_name]
                for r in range(2, max_row + 1):
                    ws.cell(row=r, column=c).number_format = fmt_1dp

    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader("", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()

try:
    df_raw = read_upload_cached(uploaded.name, file_bytes)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success(f"Loaded {len(df_raw):,} rows and {len(df_raw.columns):,} columns.")

with st.form("fao_form"):
    q1 = st.text_input(
        "Are there fields which directly alter the Cost of Attendance? Examples of fields which might directly impact overall COA include Cost Type, Commuter Status or having one of a specific group of majors where premium tuition is charged. If so, please list the names of those fields as they appear in your file, separated by a comma. If there are none, enter None."
    )
    q2 = st.text_input(
        "Is there a preferred measure of ability to pay other than HDEFC? If so, please list the name of that field. Otherwise, enter No."
    )
    q3 = st.text_input(
        "Is there a preferred measure of financial need other than HDNEED? If so, please list the name of that field. Otherwise, enter No."
    )
    q4 = st.text_input(
        "Is there a preferred numerical measure of Academic Achievement other than TOTPTS? If so, please enter the name of that field. Otherwise, enter No."
    )
    q5 = st.text_input(
        "Is there a data field which represents the value for Pell Grant after adjusting for incomplete data being received from the Partner? If so, please list the name of that field. Otherwise, enter No."
    )
    q6 = st.text_input(
        "Is there a data field which represents the value for the State's need-based award (for example, STATEFT) after adjusting for incomplete data being received from the Partner? If so, please list the name of that field. Otherwise, enter No."
    )
    q7 = st.text_input(
        "Is there a data field which represents the value for INSTMERITFT after adjusting for incomplete data being received from the Partner? If so, please enter the name of that field. Otherwise, enter No."
    )
    q8 = st.text_input("What is the name of the Partner?")

    submitted = st.form_submit_button("Run Validation Test")

if not submitted:
    st.stop()

STRAT = parse_strat_vars(q1)
BEST_EFC = "hdefc" if is_no(q2) else N(q2)
BEST_NEED = "hdneed" if is_no(q3) else N(q3)
BEST_ACAD = "totpts" if is_no(q4) else N(q4)
BEST_PELL = "pellft" if is_no(q5) else N(q5)
BEST_STATE = "stateft" if is_no(q6) else N(q6)
BEST_MERIT = "instmeritft" if is_no(q7) else N(q7)
PARTNER = (q8 or "").strip() or "PARTNER"

pos_corr = "Yes"
if BEST_ACAD != "totpts":
    pos_corr = st.selectbox(
        "Do higher values for BEST_ACADEMIC correspond to higher academic achievement? Please enter Yes or No.",
        ["Yes", "No"],
        key="pos_corr_select",
    )

cfg = ValidationConfig(
    stratification_vars=STRAT,
    best_efc=BEST_EFC,
    best_need=BEST_NEED,
    best_academic=BEST_ACAD,
    positive_iar_correlation=pos_corr,
    best_pell=BEST_PELL,
    best_state=BEST_STATE,
    best_merit=BEST_MERIT,
    partner_name=PARTNER,
)

st.subheader("Summary of Selected Fields")
st.text(
    "\n".join(
        [
            f"STRATIFICATION_VARS: {', '.join(cfg.stratification_vars) if cfg.stratification_vars else 'None'}",
            f"BEST_EFC: {cfg.best_efc}",
            f"BEST_NEED: {cfg.best_need}",
            f"BEST_ACADEMIC: {cfg.best_academic}",
            f"POSITIVE_IAR_CORRELATION: {cfg.positive_iar_correlation}",
            f"BEST_PELL: {cfg.best_pell}",
            f"BEST_STATE: {cfg.best_state}",
            f"BEST_MERIT: {cfg.best_merit}",
            f"PARTNER_NAME: {cfg.partner_name}",
        ]
    )
)

cfg_dict = asdict(cfg)

try:
    with st.spinner("Running validation..."):
        results = run_validation_cached(df_raw, cfg_dict)
        sheets = build_output_sheets(results, cfg_dict)
        excel_bytes = sheets_to_excel_bytes(sheets)
except Exception as e:
    st.error(f"Validation failed: {e}")
    st.stop()

st.download_button(
    "Download Diagnostic Test Output",
    data=excel_bytes,
    file_name=f"{cfg.partner_name}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
