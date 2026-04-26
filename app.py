"""
INTEGRATE — Intelligent Real-Time Rotating Equipment Anomaly Detection & Performance Monitoring
Team ANTEK ITS | Digital Hackathon AI/ML Hulu Migas 2026
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv("recruitment.env")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import (
    init_database, load_pump_csv, save_pump_to_db,
    get_anomaly_log, get_latest_pump_data, log_anomaly
)
from curve_engine import (
    load_pump_curves, load_compressor_curves,
    check_pump_performance, check_compressor_performance,
    PumpZone, CompressorZone
)
from alerting import get_recommended_action

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="INTEGRATE | ANTEK ITS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1a1f2e; border-radius: 8px; padding: 12px; border-left: 3px solid #00b4d8; }
    .metric-normal  { border-left: 4px solid #2ecc71 !important; }
    .metric-warning { border-left: 4px solid #f39c12 !important; }
    .metric-danger  { border-left: 4px solid #e74c3c !important; }
    .zone-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 14px; margin: 4px 0;
    }
    .zone-normal  { background: #1a4731; color: #2ecc71; }
    .zone-warning { background: #4a3000; color: #f39c12; }
    .zone-danger  { background: #4a0000; color: #e74c3c; }
    .header-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 24px; border-radius: 10px; margin-bottom: 20px;
        border: 1px solid #00b4d8;
    }
    h1 { color: #00b4d8 !important; }
    .alert-box {
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
        font-size: 14px; line-height: 1.5;
    }
    .sidebar-section { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 16px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_db():
    init_database()

init_db()

st.markdown("""
<div class="header-bar">
    <h1 style="margin:0; font-size:28px;">⚙️ INTEGRATE</h1>
    <p style="margin:4px 0 0 0; color:#8ecae6; font-size:14px;">
        Intelligent Real-Time Rotating Equipment Anomaly Detection & Performance Monitoring
        &nbsp;|&nbsp; <b>Team ANTEK ITS</b> &nbsp;|&nbsp; Digital Hackathon AI/ML Hulu Migas 2026
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ INTEGRATE")
    st.markdown("---")

    st.markdown('<p class="sidebar-section">Tipe Equipment</p>', unsafe_allow_html=True)
    equipment_type = st.selectbox("", ["Pompa (Pump)", "Kompresor (Compressor)"], label_visibility="collapsed")
    is_pump = "Pompa" in equipment_type

    st.markdown('<p class="sidebar-section">ID Equipment</p>', unsafe_allow_html=True)
    if is_pump:
        eq_id = st.selectbox("", ["P-1001A"], label_visibility="collapsed")
    else:
        eq_id = st.selectbox("", ["C-1001B"], label_visibility="collapsed")

    st.markdown('<p class="sidebar-section">Upload Data Operasi</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV / Excel (ekspor Exaquantum)", type=["csv", "xlsx"])


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_sample_data(eq_id: str, n: int = 200) -> pd.DataFrame:
    csv_path = "data/datapump_JTB/sample_pump_JTB.csv"
    if os.path.exists(csv_path):
        try:
            df = load_pump_csv(csv_path, equipment_id=eq_id)
            if len(df) >= 10:
                return df
        except Exception:
            pass
    return _generate_synthetic_pump(eq_id, n)


# ── FIX 1: _generate_synthetic_pump ─────────────────────────────────────────
# Sebelumnya: flow_rate ~370 (kBPD), suction ~42.5 psig, discharge ~310 psig
# Sekarang  : flow_rate ~12720 (BPD), suction ~161.8 psig, discharge ~1080 psig
# sesuai data DCS real dari Pump_Performance_mtd.xlsx sheet DCS Data Input
# ─────────────────────────────────────────────────────────────────────────────
def _generate_synthetic_pump(eq_id: str, n: int = 200) -> pd.DataFrame:
    """
    Generate synthetic pump data realistic sesuai DCS 340-P-1005 A/B (DMF).
    Unit: flow_rate dalam BPD, suction/discharge dalam psig.
    Konversi ke GPM/ft/hp/NPSHa dilakukan di blok perf_status.
    """
    np.random.seed(42)
    ts = pd.date_range(end=datetime.now(), periods=n, freq="15min")

    # Nilai berdasarkan data DCS real (sheet DCS Data Input Excel)
    flow      = 12720 + np.random.normal(0, 150, n)    # BPD  ← DIFIX (dulu ~370 kBPD)
    suction   = 161.8 + np.random.normal(0, 0.3, n)    # psig ← DIFIX (dulu ~42.5)
    discharge = 1080  + np.random.normal(0, 5,   n)    # psig ← DIFIX (dulu ~310)
    temperature = 95  + np.random.normal(0, 1,   n)    # degF
    current   = 145   + np.random.normal(0, 2,   n)    # Ampere
    seal      = 35    + np.random.normal(0, 0.3, n)    # psig

    # Inject anomaly di 20 titik terakhir: flow turun mendekati AOR Low
    # 12720 BPD normal, turun ~4000 BPD -> ~250 GPM (masuk AOR Low)
    flow[-20:]    -= np.linspace(0, 4000, 20)
    current[-20:] += np.linspace(0, 25, 20)
    seal[-15:]    += np.linspace(0, 8, 15)

    return pd.DataFrame({
        "timestamp"          : ts,
        "equipment_id"       : eq_id,
        "flow_rate"          : np.clip(flow,      5000, 18000),  # BPD
        "suction_pressure"   : np.clip(suction,   140,  185),    # psig
        "discharge_pressure" : np.clip(discharge,  900, 1200),   # psig
        "temperature"        : np.clip(temperature, 80,  130),   # degF
        "motor_current"      : np.clip(current,    100,  200),   # A
        "seal_pressure"      : np.clip(seal,        25,   60),   # psig
    })


@st.cache_data(ttl=60)
def load_compressor_excel(eq_id: str) -> pd.DataFrame:
    xlsx_path = "data/datacompressor_C1001B/COMPRESSOR DONGGI FIX_2 - Copy.xlsx"
    if not os.path.exists(xlsx_path):
        return _generate_synthetic_compressor(eq_id)
    try:
        import warnings
        warnings.filterwarnings("ignore")
        df = pd.read_excel(xlsx_path, sheet_name="DCS POMPA B")

        col_map = {
            "Datetime":                      "timestamp",
            "Discharge Temperature (DEGF)":  "T_discharge_F",
            "Suction Temperature (DEGF)":    "T_suction_F",
            "Suction Pressure (KG/CMÂ²)":    "suction_pressure",
            "Discharge Pressure (KG/CMÂ²)":  "discharge_pressure",
            "Flow (MMSCFD)":                 "inlet_flow",
            "Speed Compressor":              "speed_rpm",
            "Pressure Ratio":                "pressure_ratio",
            "Polytropic Head (Hp)":          "poly_head",
            "Shaft Power":                   "shaft_power",
            "poll eff ver 3":                "poly_efficiency",
        }
        df = df.rename(columns=col_map)
        keep = list(col_map.values())
        df = df[[c for c in keep if c in df.columns]].copy()

        for c in df.columns:
            if c != "timestamp":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["inlet_flow", "pressure_ratio", "shaft_power", "timestamp"])
        df = df[df["speed_rpm"] > 100]
        df["equipment_id"] = eq_id

        df["suction_pressure_abs"]   = df["suction_pressure"] + 1.033
        df["discharge_pressure_abs"] = df["discharge_pressure"] + 1.033

        surge_flow = 27.09
        df["surge_margin"] = (df["inlet_flow"] - surge_flow) / surge_flow * 100

        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.sidebar.warning(f"Gagal load Excel: {e}")
        return _generate_synthetic_compressor(eq_id)


def _generate_synthetic_compressor(eq_id: str, n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    ts = pd.date_range(end=datetime.now(), periods=n, freq="3min")

    inlet_flow  = 53.4  + np.random.normal(0, 4.3,  n)
    suc_p       = 34.3  + np.random.normal(0, 0.23, n)
    dis_p       = 61.2  + np.random.normal(0, 0.44, n)
    suc_t_F     = 95.5  + np.random.normal(0, 5.8,  n)
    dis_t_F     = 192.0 + np.random.normal(0, 3.7,  n)
    shaft_power = 1243  + np.random.normal(0, 25,   n)
    speed_rpm   = 10813 + np.random.normal(0, 200,  n)
    pr          = dis_p / suc_p
    poly_head   = 79.46 + np.random.normal(0, 1.6,  n)
    poly_eff    = 81.38 + np.random.normal(0, 0.13, n)

    inlet_flow[-15:] -= np.linspace(0, 25, 15)
    poly_head[-15:]  += np.linspace(0, 8,  15)

    surge_flow   = 27.09
    surge_margin = (inlet_flow - surge_flow) / surge_flow * 100

    return pd.DataFrame({
        "timestamp"         : ts,
        "equipment_id"      : eq_id,
        "inlet_flow"        : np.clip(inlet_flow,   9,    65),
        "suction_pressure"  : np.clip(suc_p,        33,   44),
        "discharge_pressure": np.clip(dis_p,        40,   70),
        "T_suction_F"       : np.clip(suc_t_F,      66,  115),
        "T_discharge_F"     : np.clip(dis_t_F,     142,  200),
        "shaft_power"       : np.clip(shaft_power,  800, 1350),
        "speed_rpm"         : np.clip(speed_rpm,   8500,11100),
        "pressure_ratio"    : np.clip(pr,           1.0,  1.9),
        "poly_head"         : np.clip(poly_head,    -5,   90),
        "poly_efficiency"   : np.clip(poly_eff,    80.9, 82.5),
        "surge_margin"      : np.clip(surge_margin,-10,  120),
    })


# Handle file upload
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
        df_raw.columns = df_raw.columns.str.lower().str.strip().str.replace(" ", "_")
        st.sidebar.success(f"✅ Data dimuat: {len(df_raw)} baris")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
        df_raw = None
else:
    df_raw = None

if df_raw is not None and len(df_raw) > 0:
    df = df_raw.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="15min")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
else:
    if is_pump:
        df = load_sample_data(eq_id)
    else:
        df = load_compressor_excel(eq_id)

df = df.sort_values("timestamp").reset_index(drop=True)

# ──────────────────────────────────────────────
# GLOBAL DATE RANGE FILTER
# ──────────────────────────────────────────────

_df_full = df.copy()
_ts_full = pd.to_datetime(_df_full["timestamp"])
_d_min   = _ts_full.min().date()
_d_max   = _ts_full.max().date()

if st.session_state.get("_g_eq") != eq_id:
    st.session_state["_g_start"] = _d_min
    st.session_state["_g_end"]   = _d_max
    st.session_state["_g_eq"]    = eq_id

if "_g_start" not in st.session_state:
    st.session_state["_g_start"] = _d_min
    st.session_state["_g_end"]   = _d_max

with st.sidebar:
    st.markdown("---")
    st.markdown('<p class="sidebar-section">Filter Rentang Data</p>', unsafe_allow_html=True)
    with st.form("global_date_form"):
        _g_start = st.date_input(
            "Dari", value=st.session_state["_g_start"],
            min_value=_d_min, max_value=_d_max)
        _g_end = st.date_input(
            "Sampai", value=st.session_state["_g_end"],
            min_value=_d_min, max_value=_d_max)
        _apply_btn = st.form_submit_button(
            "▶ Terapkan Filter", use_container_width=True, type="primary")
    if _apply_btn:
        st.session_state["_g_start"] = _g_start
        st.session_state["_g_end"]   = _g_end
        st.rerun()

    _n_sel = int(((_ts_full.dt.date >= st.session_state["_g_start"]) &
                  (_ts_full.dt.date <= st.session_state["_g_end"])).sum())
    st.caption(f"Terpilih: **{_n_sel:,}** dari {len(_df_full):,} titik data")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Info Sistem</p>', unsafe_allow_html=True)
    st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    st.caption(f"📊 Database: {os.getenv('DB_PATH', 'integrate.db')}")
    st.caption("v1.0.0 — ANTEK ITS")

_g_mask = ((_ts_full.dt.date >= st.session_state["_g_start"]) &
           (_ts_full.dt.date <= st.session_state["_g_end"]))
df = _df_full[_g_mask].copy().reset_index(drop=True)

if len(df) == 0:
    st.warning("Tidak ada data dalam rentang tanggal yang dipilih. Ubah filter di sidebar kiri.")
    st.stop()

# ──────────────────────────────────────────────
# ANOMALY DETECTION
# ──────────────────────────────────────────────

def compute_mae_heuristic(df: pd.DataFrame, features: list) -> pd.Series:
    numeric      = df[features].select_dtypes(include=np.number)
    rolling_mean = numeric.rolling(10, min_periods=1).mean()
    mae          = (numeric - rolling_mean).abs().mean(axis=1)
    return mae


if is_pump:
    pump_features_all = [
        "flow_rate", "suction_pressure", "discharge_pressure",
        "temperature", "motor_current", "seal_pressure",
    ]
    avail_features = [c for c in pump_features_all if c in df.columns]
else:
    comp_features  = [
        "inlet_flow", "suction_pressure", "discharge_pressure",
        "pressure_ratio", "poly_head", "shaft_power", "poly_efficiency", "speed_rpm",
    ]
    avail_features = [c for c in comp_features if c in df.columns]

if avail_features:
    df["mae"]        = compute_mae_heuristic(df, avail_features)
    mae_mean         = df["mae"].mean()
    mae_std          = df["mae"].std()
    threshold        = mae_mean + 2.33 * mae_std
    df["is_anomaly"] = df["mae"] > threshold
    df["threshold"]  = threshold
else:
    df["mae"] = 0.0; df["is_anomaly"] = False; df["threshold"] = 0.0; threshold = 0.0

latest     = df.iloc[-1]
n_anomaly  = df["is_anomaly"].sum()
anomaly_pct = n_anomaly / len(df) * 100

# ──────────────────────────────────────────────
# PERFORMANCE CURVE CHECK
# ──────────────────────────────────────────────

pump_curves = load_pump_curves()
comp_curves = load_compressor_curves()

# ── FIX 2: Konversi unit DCS → GPM/ft/hp/NPSHa ──────────────────────────────
# Sebelumnya: flow_bpd = float(latest.get("flow_rate", 12720)) tanpa auto-detect
# Sekarang  : auto-detect unit (BPD vs kBPD) berdasarkan median nilai flow_rate
#
# Formula dari Pump_Performance_mtd.xlsx sheet Summary:
#   Flow (GPM)  = Flow_BPD × 42 / 1440
#   Head (ft)   = (Pd_psig − Ps_psig) / (0.4335 × SG)
#   BHP  (hp)   = Flow_GPM × (Pd_psig − Ps_psig) / (1715 × η)
#   Ps   (psia) = Ps_psig + 14.7
#   NPSHa (ft)  = (Ps_psia − Pv_psia) / (0.4335 × SG)
# ─────────────────────────────────────────────────────────────────────────────
if is_pump and "flow_rate" in df.columns:
    # Fluid properties dari pump_curves
    fluid = pump_curves.get("fluid", {})
    sg    = fluid.get("specific_gravity",
            pump_curves.get("npsh", {}).get("specific_gravity", 1.084))
    pv    = fluid.get("vapor_pressure_psia",
            pump_curves.get("npsh", {}).get("vapor_pressure_psia", 1.02))
    eta   = fluid.get("pump_efficiency", 0.686)

    # Auto-detect unit flow_rate:
    # - BPD (DCS real / synthetic baru): nilai ~12720, median > 2000
    # - kBPD (format lain): nilai ~370, median < 2000
    median_flow = float(df["flow_rate"].median())
    if median_flow < 2000:
        # kBPD → GPM: kBPD × 1000 × 42 / 1440
        flow_val = float(latest["flow_rate"]) * 1000.0 * 42.0 / 1440.0
    else:
        # BPD → GPM: BPD × 42 / 1440
        flow_val = float(latest["flow_rate"]) * 42.0 / 1440.0

    ps_psig   = float(latest.get("suction_pressure",    161.8))
    pd_psig   = float(latest.get("discharge_pressure", 1080.0))

    head_val  = (pd_psig - ps_psig) / (0.4335 * sg)
    power_val = (flow_val * (pd_psig - ps_psig)) / (1715.0 * eta)
    npsha_val = (ps_psig + 14.7 - pv) / (0.4335 * sg)

    pump_curves["npsh"]["npsha"] = npsha_val

    perf_status = check_pump_performance(
        eq_id, flow_val, head_val, power_val, npsha_val, pump_curves
    )
else:
    perf_status = None

if not is_pump and "inlet_flow" in df.columns:
    flow_val_c  = float(latest.get("inlet_flow", 53.4))
    pr_val      = float(latest.get("pressure_ratio", 1.767))
    ph_val      = float(latest.get("poly_head", latest.get("polytropic_head", 79.5)))
    sp_val      = float(latest.get("shaft_power", 1243))
    pe_val      = float(latest.get("poly_efficiency", 81.38))
    spd_val     = float(latest.get("speed_rpm", 10813))
    perf_status = check_compressor_performance(
        eq_id, flow_val_c, pr_val, ph_val, sp_val, pe_val, spd_val, comp_curves)
elif is_pump:
    pass
else:
    perf_status = None

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Real-Time",
    "📈 Kurva Performa",
    "🤖 Anomali ML",
    "📋 Log & Riwayat",
    "🏭 Implementasi"
])

# ════════════════════════════════════════════════
# TAB 1 — REAL-TIME DASHBOARD
# ════════════════════════════════════════════════

with tab1:
    if latest.get("is_anomaly", False):
        st.error(f"⚠️ **ANOMALI TERDETEKSI** pada {eq_id} — MAE={latest['mae']:.4f} (Threshold={threshold:.4f})")
    else:
        st.success(f"✅ **Operasi Normal** — {eq_id} | Data terakhir: {pd.to_datetime(latest.get('timestamp', datetime.now())).strftime('%d/%m/%Y %H:%M')}")

    if is_pump:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        cols = [col1, col2, col3, col4, col5, col6]
        metrics = [
            ("Flow Rate", f"{latest.get('flow_rate', 0):.1f}", "BPD",  12720, 150),
            ("Suction P", f"{latest.get('suction_pressure', 0):.1f}",  "psig", 161.8, 2),
            ("Discharge P",f"{latest.get('discharge_pressure', 0):.1f}","psig",1080,  10),
            ("Temperatur", f"{latest.get('temperature', 0):.1f}",       "°F",   95,   5),
            ("Motor I",    f"{latest.get('motor_current', 0):.1f}",     "A",   145,  15),
            ("Seal P",     f"{latest.get('seal_pressure', 0):.1f}",     "psig", 35,   5),
        ]
    else:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        cols = [col1, col2, col3, col4, col5, col6]
        metrics = [
            ("Inlet Flow",    f"{latest.get('inlet_flow', 0):.2f}",      "MMSCFD", 53.4,  4.3),
            ("Suction P",     f"{latest.get('suction_pressure', 0):.3f}", "kg/cm²", 34.29, 0.23),
            ("Discharge P",   f"{latest.get('discharge_pressure', 0):.3f}","kg/cm²",61.24, 0.44),
            ("Pressure Ratio",f"{latest.get('pressure_ratio', 0):.4f}",   "–",      1.763, 0.017),
            ("Shaft Power",   f"{latest.get('shaft_power', 0):.1f}",      "kW",     1243,  25),
            ("Poly Head",     f"{latest.get('poly_head', latest.get('polytropic_head',0)):.2f}", "kJ/kg", 79.46, 1.6),
        ]

    for col, (label, value, unit, nominal, tol) in zip(cols, metrics):
        try:
            val_num = float(value.replace(",", "."))
            delta   = val_num - nominal
            col.metric(label=f"{label} ({unit})", value=value, delta=f"{delta:+.1f}")
        except Exception:
            col.metric(label=f"{label} ({unit})", value=value)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📉 Tren Parameter Operasi")
        if is_pump:
            primary_cols = [c for c in ["flow_rate", "motor_current", "seal_pressure"] if c in df.columns]
        else:
            primary_cols = [c for c in ["inlet_flow", "shaft_power", "poly_head", "speed_rpm"] if c in df.columns]

        fig_ts = go.Figure()
        colors = ["#00b4d8", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        for i, col in enumerate(primary_cols):
            fig_ts.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col],
                name=col.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=1.5),
                mode="lines"
            ))
        fig_ts.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=-0.2),
            xaxis_title="Waktu", yaxis_title="Nilai"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_right:
        st.subheader("🔍 MAE — Reconstruction Error")
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=df["timestamp"], y=df["mae"],
            name="MAE", line=dict(color="#00b4d8", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.15)"
        ))
        fig_mae.add_hline(
            y=threshold, line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Threshold={threshold:.4f}",
            annotation_position="bottom right"
        )
        df_anom = df[df["is_anomaly"]]
        if len(df_anom) > 0:
            fig_mae.add_trace(go.Scatter(
                x=df_anom["timestamp"], y=df_anom["mae"],
                name="Anomali", mode="markers",
                marker=dict(color="#e74c3c", size=6, symbol="x")
            ))
        fig_mae.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=-0.2),
            xaxis_title="Waktu", yaxis_title="MAE"
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    st.subheader("🔄 Tren Tekanan")
    p_cols = [c for c in ["suction_pressure", "discharge_pressure"] if c in df.columns]
    if p_cols:
        fig_p = go.Figure()
        for i, col in enumerate(p_cols):
            fig_p.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col],
                name=col.replace("_", " ").title(),
                line=dict(color=colors[i], width=1.5)
            ))
        fig_p.update_layout(
            template="plotly_dark", height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.3)
        )
        st.plotly_chart(fig_p, use_container_width=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("⚡ Status Anomali Saat Ini")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=float(latest.get("mae", 0)),
            delta={"reference": threshold, "valueformat": ".4f"},
            title={"text": "MAE Score", "font": {"color": "white"}},
            number={"valueformat": ".4f", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, threshold * 2], "tickcolor": "white"},
                "bar": {"color": "#00b4d8"},
                "bgcolor": "#1a1f2e",
                "bordercolor": "#333",
                "steps": [
                    {"range": [0, threshold * 0.7], "color": "#1a4731"},
                    {"range": [threshold * 0.7, threshold], "color": "#4a3000"},
                    {"range": [threshold, threshold * 2], "color": "#4a0000"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 3},
                    "thickness": 0.8,
                    "value": threshold
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark", height=260,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="#0e1117", font_color="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.subheader("📊 Distribusi Anomali")
        anom_counts = df["is_anomaly"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Anomali"],
            values=[anom_counts.get(False, 0), anom_counts.get(True, 0)],
            hole=0.55,
            marker_colors=["#2ecc71", "#e74c3c"],
            textinfo="label+percent"
        ))
        fig_pie.update_layout(
            template="plotly_dark", height=260,
            margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="#0e1117", font_color="white",
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 2 — PERFORMANCE CURVE
# ════════════════════════════════════════════════

with tab2:
    st.subheader(f"📈 Analisis Kurva Performa — {eq_id}")

    if perf_status is not None:
        zone_str = perf_status.zone.value
        if any(x in zone_str for x in ["Surge", "Overload", "Min Flow", "Cavitation"]) or \
           "CAVITATION" in zone_str.upper():
            badge_cls = "zone-danger"
        elif "AOR" in zone_str or "Protection" in zone_str:
            badge_cls = "zone-warning"
        else:
            badge_cls = "zone-normal"

        st.markdown(f'<span class="zone-badge {badge_cls}">🔹 Zona: {zone_str}</span>',
                    unsafe_allow_html=True)

        for msg in perf_status.alert_messages:
            if "⛔" in msg or "🚨" in msg:
                st.error(msg)
            elif "⚠️" in msg:
                st.warning(msg)
            else:
                st.success(msg)

        st.markdown("---")

    st.info(
        f"Menampilkan data: **{st.session_state['_g_start'].strftime('%d %b %Y')}** "
        f"s.d. **{st.session_state['_g_end'].strftime('%d %b %Y')}** "
        f"({len(df):,} titik). Ubah rentang melalui filter di sidebar kiri.",
        icon="📅")
    st.markdown("---")

    # ── FIX 3: Plot Tab 2 Pump ───────────────────────────────────────────────
    # Sebelumnya: hanya 1 kurva (Rated), actual point salah posisi (flow ~10 GPM)
    # Sekarang  : 3 kurva (Max/Rated/Min diameter), actual point dari perf_status
    #             yang sudah benar (flow ~371 GPM, head ~1954 ft, dll)
    # ─────────────────────────────────────────────────────────────────────────
    if is_pump:
        hf  = pump_curves["head_flow"]
        npsh_c = pump_curves["npsh"]
        pwr = pump_curves["power"]

        rated_flow  = hf["rated_flow"]                          # 600.2 GPM (BEP)
        por_min_gpm = hf["por_min_flow"] / 100 * rated_flow     # ~420 GPM
        por_max_gpm = hf["por_max_flow"] / 100 * rated_flow     # ~711.6 GPM
        aor_min_gpm = hf["aor_min_flow"] / 100 * rated_flow     # ~115 GPM

        flow_range = np.linspace(0, 800, 500)

        # 3 kurva head — gunakan coeff masing-masing diameter dari curves
        head_max_arr   = np.polyval(hf.get("coeff_max_diameter",   hf["coefficients"]), flow_range)
        head_rated_arr = np.polyval(hf.get("coeff_rated_diameter", hf["coefficients"]), flow_range)
        head_min_arr   = np.polyval(hf.get("coeff_min_diameter",   hf["coefficients"]), flow_range)

        # Actual point: ambil dari perf_status (sudah dalam unit yang benar)
        if perf_status:
            op_flow  = perf_status.flow    # GPM
            op_head  = perf_status.head    # ft
            op_bhp   = perf_status.power   # hp
            op_npsha = perf_status.npsha   # ft
        else:
            op_flow  = 371.0
            op_head  = 1954.0
            op_bhp   = 289.6
            op_npsha = 373.3

        # ════ PLOT 1 — HEAD vs FLOW ════
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("**Kurva H-Q Pompa — API 610 (Head vs Flow)**")
            fig_hq = go.Figure()

            # Shading AOR (oranye muda)
            fig_hq.add_vrect(
                x0=aor_min_gpm, x1=por_min_gpm,
                fillcolor="rgba(243,156,18,0.10)", line_width=0,
                annotation_text="AOR", annotation_position="top left",
                annotation_font=dict(color="#f39c12", size=10))
            # Shading POR (hijau muda)
            fig_hq.add_vrect(
                x0=por_min_gpm, x1=por_max_gpm,
                fillcolor="rgba(46,204,113,0.12)", line_width=0,
                annotation_text="POR", annotation_position="top left",
                annotation_font=dict(color="#2ecc71", size=10))

            # 3 kurva diameter (warna sesuai Excel: Blue=Max, Gray=Rated, Orange=Min)
            fig_hq.add_trace(go.Scatter(
                x=flow_range, y=head_max_arr,
                name="Max Diameter (10.79\")",
                mode="lines", line=dict(color="#1f77b4", width=2.5)))
            fig_hq.add_trace(go.Scatter(
                x=flow_range, y=head_rated_arr,
                name="Rated Diameter (10.35\")",
                mode="lines", line=dict(color="#7f7f7f", width=2.5)))
            fig_hq.add_trace(go.Scatter(
                x=flow_range, y=head_min_arr,
                name="Min Diameter (9.65\")",
                mode="lines", line=dict(color="#ff7f0e", width=2.5)))

            # Garis vertikal batas (dashed — sesuai Excel chart)
            fig_hq.add_vline(
                x=aor_min_gpm,
                line=dict(color="#17becf", width=2, dash="dash"),
                annotation_text=f"Min Flow<br>{aor_min_gpm:.0f} GPM",
                annotation_position="top right",
                annotation_font=dict(color="#17becf", size=9))
            fig_hq.add_vline(
                x=por_min_gpm,
                line=dict(color="#2ca02c", width=2, dash="dash"),
                annotation_text=f"Lower POR<br>{por_min_gpm:.0f} GPM",
                annotation_position="top right",
                annotation_font=dict(color="#2ca02c", size=9))
            fig_hq.add_vline(
                x=por_max_gpm,
                line=dict(color="#1a1aff", width=2, dash="dash"),
                annotation_text=f"Upper POR<br>{por_max_gpm:.0f} GPM",
                annotation_position="top right",
                annotation_font=dict(color="#1a1aff", size=9))

            # Actual Point DCS
            fig_hq.add_trace(go.Scatter(
                x=[op_flow], y=[op_head],
                name="Actual Point - DCS",
                mode="markers",
                marker=dict(color="red", size=14, symbol="diamond",
                            line=dict(color="white", width=2))))

            fig_hq.update_layout(
                template="plotly_dark", height=420,
                title=dict(text="Head vs Flow", x=0.5,
                           font=dict(color="white", size=13)),
                xaxis=dict(title="Flow (GPM)", range=[0, 850], dtick=100,
                           gridcolor="#2a2f3e"),
                yaxis=dict(title="Head (ft)", range=[1000, 2500], dtick=200,
                           gridcolor="#2a2f3e"),
                hovermode="x unified",
                legend=dict(orientation="v", x=1.01, y=1,
                            font=dict(color="white", size=9),
                            bgcolor="rgba(26,31,46,0.9)",
                            bordercolor="#444", borderwidth=1),
                margin=dict(l=10, r=175, t=50, b=10),
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e")
            st.plotly_chart(fig_hq, use_container_width=True)

        # ════ PLOT 2 — NPSHa vs NPSHr ════
        with col_c2:
            st.markdown("**NPSHa vs NPSHr — ANSI/HI 9.6.1**")

            npshr_arr = np.clip(
                np.polyval(npsh_c["npshr_coefficients"], flow_range), 0, None)

            fig_npsh = go.Figure()
            fig_npsh.add_trace(go.Scatter(
                x=flow_range, y=npshr_arr,
                name="NPSHR (Required)",
                mode="lines", line=dict(color="#1f77b4", width=2.5)))
            fig_npsh.add_trace(go.Scatter(
                x=flow_range, y=1.3 * npshr_arr,
                name="1.3 × NPSHR (Warning)",
                mode="lines", line=dict(color="orange", width=1.5, dash="dot")))
            fig_npsh.add_trace(go.Scatter(
                x=[op_flow], y=[op_npsha],
                name="NPSHa Aktual",
                mode="markers",
                marker=dict(color="orange", size=12, symbol="circle",
                            line=dict(color="white", width=1))))

            fig_npsh.update_layout(
                template="plotly_dark", height=420,
                title=dict(text="NPSH Available vs NPSH Required", x=0.5,
                           font=dict(color="white", size=13)),
                xaxis=dict(title="Flow (GPM)", range=[0, 850], dtick=100,
                           gridcolor="#2a2f3e"),
                yaxis=dict(title="NPSH (ft)", range=[0, 450], dtick=50,
                           gridcolor="#2a2f3e"),
                hovermode="x unified",
                legend=dict(orientation="v", x=1.01, y=1,
                            font=dict(color="white", size=9),
                            bgcolor="rgba(26,31,46,0.9)",
                            bordercolor="#444", borderwidth=1),
                margin=dict(l=10, r=175, t=50, b=10),
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e")
            st.plotly_chart(fig_npsh, use_container_width=True)

        # ════ PLOT 3 — Power vs Flow ════
        st.markdown("**Power vs Flow (BHP)**")

        power_arr = np.clip(
            np.polyval(pwr["coefficients"], flow_range), 0, None)

        fig_pw = go.Figure()
        fig_pw.add_trace(go.Scatter(
            x=flow_range, y=power_arr,
            name="Power Curve",
            mode="lines", line=dict(color="#9467bd", width=2.5),
            fill="tozeroy", fillcolor="rgba(148,103,189,0.08)"))
        fig_pw.add_hline(
            y=pwr["rated_power"] * 1.10,
            line=dict(color="red", width=1.5, dash="dash"),
            annotation_text=f"110% Rated ({pwr['rated_power']*1.10:.0f} hp) — Overload Limit",
            annotation_position="top right",
            annotation_font=dict(color="red", size=9))
        fig_pw.add_vline(x=por_min_gpm,
            line=dict(color="#2ca02c", width=1.5, dash="dash"),
            annotation_text="Lower POR",
            annotation_position="top right",
            annotation_font=dict(color="#2ca02c", size=9))
        fig_pw.add_vline(x=por_max_gpm,
            line=dict(color="#1a1aff", width=1.5, dash="dash"),
            annotation_text="Upper POR",
            annotation_position="top right",
            annotation_font=dict(color="#1a1aff", size=9))
        fig_pw.add_trace(go.Scatter(
            x=[op_flow], y=[op_bhp],
            name="BHP Aktual",
            mode="markers",
            marker=dict(color="red", size=14, symbol="diamond",
                        line=dict(color="white", width=2))))

        fig_pw.update_layout(
            template="plotly_dark", height=350,
            title=dict(text="Power vs Flow", x=0.5,
                       font=dict(color="white", size=13)),
            xaxis=dict(title="Flow (GPM)", range=[0, 850], dtick=100,
                       gridcolor="#2a2f3e"),
            yaxis=dict(title="BHP (hp)", gridcolor="#2a2f3e"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(color="white")),
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e")
        st.plotly_chart(fig_pw, use_container_width=True)

    # ── COMPRESSOR: 4 Performance Curve Charts (tidak berubah) ──
    else:
        _fitted_path = "curves/compressor_C1001B_fitted.json"
        _fitted = {}
        if os.path.exists(_fitted_path):
            with open(_fitted_path) as _f:
                _fitted = json.load(_f)

        flow_range  = np.linspace(26, 68, 300)
        n_sc        = min(500, len(df))
        df_sc       = df.tail(n_sc)
        ops         = comp_curves.get("operating_stats", {})
        fpr         = comp_curves["flow_pressure_ratio"]
        surge_flow  = fpr["surge_flow"]
        prot_flow   = fpr["protection_flow"]
        rated_power = comp_curves["shaft_power"].get("rated_power", 1280.0)

        _flow  = perf_status.inlet_flow      if perf_status else 53.43
        _pr    = perf_status.pressure_ratio  if perf_status else 1.763
        _ph    = perf_status.polytropic_head if perf_status else 79.46
        _sp    = perf_status.shaft_power     if perf_status else 1243.0
        _pe    = perf_status.poly_efficiency if perf_status else 81.38
        _spd   = perf_status.speed_rpm       if perf_status else 10813.0
        _dis_p = float(latest.get("discharge_pressure", 57.0))
        _suc_p = float(latest.get("suction_pressure", 33.5))

        _zone_colors = {
            "Z1": "#e74c3c", "Z2": "#e67e22", "Z3": "#2ecc71",
            "Z4": "#27ae60", "Z5": "#f39c12", "Z6": "#e67e22",
            "Z7": "#e74c3c", "Z8": "#c0392b"
        }
        _pt_color = _zone_colors.get(
            perf_status.zone.value[:2] if perf_status else "Z5", "#f39c12")

        _speed_cfg = [
            ("speed_A", "A — 13068 RPM", "#9b59b6"),
            ("speed_B", "B — 12446 RPM", "#3498db"),
            ("speed_C", "C — 11201 RPM", "#2ecc71"),
            ("speed_D", "D — 9957 RPM",  "#f39c12"),
            ("speed_E", "E — 8712 RPM",  "#95a5a6"),
        ]

        def _add_op(fig, x, y, label, color=None, size=15):
            fig.add_trace(go.Scatter(
                x=[x], y=[y], name=label, mode="markers",
                marker=dict(color=color or _pt_color, size=size, symbol="diamond",
                            line=dict(color="white", width=2))
            ))

        _ph_col = "poly_head" if "poly_head" in df_sc.columns else "polytropic_head"

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**① Polytropic Head vs Standard Inlet Flow**")
            fig1 = go.Figure()
            for key, label, color in _speed_cfg:
                if key in _fitted:
                    _xmin = _fitted[key]["x_min"]
                    _xmax = _fitted[key]["x_max"]
                    _yv   = np.polyval(_fitted[key]["coeff"], flow_range)
                    _yv   = np.where((flow_range >= _xmin) & (flow_range <= _xmax), _yv, np.nan)
                    fig1.add_trace(go.Scatter(x=flow_range, y=_yv, name=label,
                                             line=dict(color=color, width=1.8, dash="dot"),
                                             mode="lines"))
            for _k, _ln, _lc, _ld in [
                ("surge_line",      "Surge Line",      "#e74c3c", "solid"),
                ("protection_line", "Protection Line", "#f39c12", "dash")
            ]:
                if _k in _fitted:
                    _yv = np.polyval(_fitted[_k]["coeff"], flow_range)
                    _yv = np.where(
                        (flow_range >= _fitted[_k]["x_min"]) &
                        (flow_range <= _fitted[_k]["x_max"]), _yv, np.nan)
                    fig1.add_trace(go.Scatter(x=flow_range, y=_yv, name=_ln,
                                             line=dict(color=_lc, width=2.2, dash=_ld),
                                             mode="lines"))
            if _ph_col in df_sc.columns and "inlet_flow" in df_sc.columns:
                fig1.add_trace(go.Scatter(x=df_sc["inlet_flow"], y=df_sc[_ph_col],
                                         name="Data DCS", mode="markers",
                                         marker=dict(color="#00b4d8", size=3, opacity=0.35)))
            _add_op(fig1, _flow, _ph,
                    f"Titik Operasi ({perf_status.zone.value if perf_status else ''})")
            fig1.update_layout(template="plotly_dark", height=390,
                               xaxis_title="Std Inlet Flow (MMSCFD)",
                               yaxis_title="Polytropic Head (kJ/kg)",
                               xaxis=dict(range=[25, 70]), yaxis=dict(range=[25, 165]),
                               margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=-0.38, font=dict(size=9)))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown("**② Pressure Ratio vs Standard Inlet Flow**")
            fig2 = go.Figure()
            for _k, _lbl, _lc in _speed_cfg:
                _fk = "pr_" + _k
                if _fk in _fitted:
                    _yv = np.polyval(_fitted[_fk]["coeff"], flow_range)
                    _yv = np.where(
                        (flow_range >= _fitted[_fk]["x_min"]) &
                        (flow_range <= _fitted[_fk]["x_max"]), _yv, np.nan)
                    fig2.add_trace(go.Scatter(x=flow_range, y=_yv, name=_lbl,
                                             line=dict(color=_lc, width=1.8, dash="dot"),
                                             mode="lines"))
            for _fk, _ln, _lc, _ld in [
                ("pr_surge_line",      "Surge Line",      "#e74c3c", "solid"),
                ("pr_protection_line", "Protection Line", "#f39c12", "dash")
            ]:
                if _fk in _fitted:
                    _yv = np.polyval(_fitted[_fk]["coeff"], flow_range)
                    _yv = np.where(
                        (flow_range >= _fitted[_fk]["x_min"]) &
                        (flow_range <= _fitted[_fk]["x_max"]), _yv, np.nan)
                    fig2.add_trace(go.Scatter(x=flow_range, y=_yv, name=_ln,
                                             line=dict(color=_lc, width=2.2, dash=_ld),
                                             mode="lines"))
            if "pressure_ratio" in df_sc.columns and "inlet_flow" in df_sc.columns:
                fig2.add_trace(go.Scatter(x=df_sc["inlet_flow"], y=df_sc["pressure_ratio"],
                                         name="Data DCS", mode="markers",
                                         marker=dict(color="#9b59b6", size=3, opacity=0.35)))
            _add_op(fig2, _flow, _pr, f"Titik Operasi (PR={_pr:.4f})")
            fig2.update_layout(template="plotly_dark", height=390,
                               xaxis_title="Std Inlet Flow (MMSCFD)",
                               yaxis_title="Pressure Ratio",
                               xaxis=dict(range=[25, 70]),
                               margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=-0.38, font=dict(size=9)))
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**③ Discharge Pressure vs Standard Inlet Flow**")
            _design_dis = round((_suc_p + 1.033) * fpr["rated_pressure_ratio"] - 1.033, 1)
            _dis_mean   = round((_suc_p + 1.033) * ops.get("pr_mean", 1.763) - 1.033, 2)
            fig3 = go.Figure()
            for _k, _lbl, _lc in _speed_cfg:
                _fk = "dp_" + _k
                if _fk in _fitted:
                    _yv = np.polyval(_fitted[_fk]["coeff"], flow_range)
                    _yv = np.where(
                        (flow_range >= _fitted[_fk]["x_min"]) &
                        (flow_range <= _fitted[_fk]["x_max"]), _yv, np.nan)
                    fig3.add_trace(go.Scatter(x=flow_range, y=_yv, name=_lbl,
                                             line=dict(color=_lc, width=1.8, dash="dot"),
                                             mode="lines"))
            for _fk, _ln, _lc, _ld in [
                ("dp_surge_line",      "Surge Line",      "#e74c3c", "solid"),
                ("dp_protection_line", "Protection Line", "#f39c12", "dash")
            ]:
                if _fk in _fitted:
                    _yv = np.polyval(_fitted[_fk]["coeff"], flow_range)
                    _yv = np.where(
                        (flow_range >= _fitted[_fk]["x_min"]) &
                        (flow_range <= _fitted[_fk]["x_max"]), _yv, np.nan)
                    fig3.add_trace(go.Scatter(x=flow_range, y=_yv, name=_ln,
                                             line=dict(color=_lc, width=2.2, dash=_ld),
                                             mode="lines"))
            if "discharge_pressure" in df_sc.columns and "inlet_flow" in df_sc.columns:
                fig3.add_trace(go.Scatter(x=df_sc["inlet_flow"], y=df_sc["discharge_pressure"],
                                         name="Data DCS", mode="markers",
                                         marker=dict(color="#e67e22", size=3, opacity=0.35)))
            _add_op(fig3, _flow, _dis_p, f"Titik Operasi ({_dis_p:.2f} kg/cm2)")
            fig3.update_layout(template="plotly_dark", height=390,
                               xaxis_title="Std Inlet Flow (MMSCFD)",
                               yaxis_title="Discharge Pressure (kg/cm2-g)",
                               xaxis=dict(range=[25, 70]),
                               margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=-0.38, font=dict(size=9)))
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            st.markdown("**④ Shaft Power vs Standard Inlet Flow**")
            _sp_mean = ops.get("shaft_power_mean", 1243.0)
            _sp_p90  = ops.get("shaft_power_p90",  1262.1)
            _sp_p99  = ops.get("shaft_power_p99",  1278.2)
            fig4 = go.Figure()
            for _k, _lbl, _lc in _speed_cfg:
                _fk = "sp_" + _k
                if _fk in _fitted:
                    _yv = np.polyval(_fitted[_fk]["coeff"], flow_range)
                    _yv = np.where(
                        (flow_range >= _fitted[_fk]["x_min"]) &
                        (flow_range <= _fitted[_fk]["x_max"]), _yv, np.nan)
                    fig4.add_trace(go.Scatter(x=flow_range, y=_yv, name=_lbl,
                                             line=dict(color=_lc, width=1.8, dash="dot"),
                                             mode="lines"))
            fig4.add_hline(y=_sp_mean, line_color="#2ecc71", line_dash="dot",
                          annotation_text=f"Mean {_sp_mean:.0f} kW",
                          annotation_font_size=9, annotation_font_color="#2ecc71")
            fig4.add_hline(y=rated_power, line_color="#e67e22", line_dash="dash",
                          annotation_text=f"Rated {rated_power:.0f} kW",
                          annotation_font_size=9, annotation_font_color="#e67e22")
            fig4.add_hline(y=rated_power * 1.1, line_color="#e74c3c", line_dash="dash",
                          annotation_text="110% Rated",
                          annotation_font_size=9, annotation_font_color="#e74c3c")
            if "shaft_power" in df_sc.columns and "inlet_flow" in df_sc.columns:
                fig4.add_trace(go.Scatter(x=df_sc["inlet_flow"], y=df_sc["shaft_power"],
                                         name="Data DCS", mode="markers",
                                         marker=dict(color="#f39c12", size=3, opacity=0.35)))
            _add_op(fig4, _flow, _sp, f"Titik Operasi ({_sp:.0f} kW)")
            fig4.update_layout(template="plotly_dark", height=390,
                               xaxis_title="Std Inlet Flow (MMSCFD)",
                               yaxis_title="Shaft Power (kW)",
                               xaxis=dict(range=[25, 70]),
                               margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=-0.38, font=dict(size=9)))
            st.plotly_chart(fig4, use_container_width=True)

        # ══ 4 Individual Analyses & Kesimpulan (tidak berubah) ══
        st.markdown("---")
        st.markdown("### Analisis Per Grafik")

        _ph_mean   = ops.get("poly_head_mean", 79.46)
        _ph_p25    = ops.get("poly_head_p25",  79.24)
        _ph_p75    = ops.get("poly_head_p75",  80.20)
        _pr_mean_a = ops.get("pr_mean", 1.763)
        _pr_p25_a  = ops.get("pr_p25",  1.755)
        _pr_p75_a  = ops.get("pr_p75",  1.773)
        _sp_mean_a = ops.get("shaft_power_mean", 1243.0)
        _sp_p90_a  = ops.get("shaft_power_p90",  1262.1)
        _sp_p99_a  = ops.get("shaft_power_p99",  1278.2)

        if _ph_p25 <= _ph <= _ph_p75:   _ph_st = "NORMAL";  _ph_cl = "#2ecc71"
        elif _ph > _ph_p75:             _ph_st = "TINGGI";  _ph_cl = "#f39c12"
        elif _ph > 70:                  _ph_st = "RENDAH";  _ph_cl = "#f39c12"
        else:                           _ph_st = "KRITIS";  _ph_cl = "#e74c3c"

        if _pr_p25_a <= _pr <= _pr_p75_a:    _pr_st = "NORMAL"; _pr_cl = "#2ecc71"
        elif _pr > _pr_p75_a + 0.01:         _pr_st = "TINGGI"; _pr_cl = "#f39c12"
        else:                                 _pr_st = "RENDAH"; _pr_cl = "#f39c12"

        _dis_dev = _dis_p - _dis_mean
        if abs(_dis_dev) <= 2.0:  _dis_st = "NORMAL"; _dis_cl = "#2ecc71"
        elif _dis_dev > 2.0:      _dis_st = "TINGGI"; _dis_cl = "#f39c12"
        else:                     _dis_st = "RENDAH"; _dis_cl = "#f39c12"

        _sp_load_pct = (_sp / rated_power) * 100
        if _sp <= _sp_p90_a:        _sp_st = "NORMAL";           _sp_cl = "#2ecc71"
        elif _sp <= _sp_p99_a:      _sp_st = "TINGGI";           _sp_cl = "#f39c12"
        elif _sp <= rated_power:    _sp_st = "MENDEKATI RATED";  _sp_cl = "#e67e22"
        else:                       _sp_st = "OVERLOAD";         _sp_cl = "#e74c3c"

        _ph_desc = (
            "Polytropic head dalam rentang normal operasi P25-P75."
            if _ph_st == "NORMAL" else
            f"Hp di atas P75 ({_ph_p75:.2f} kJ/kg). Beban/speed lebih tinggi dari tipikal."
            if _ph_st == "TINGGI" else
            f"Hp di bawah P25 ({_ph_p25:.2f} kJ/kg). Evaluasi speed dan kondisi gas masuk."
            if _ph_st == "RENDAH" else
            "Hp sangat rendah (< 70 kJ/kg). KRITIS — cek sensor dan speed kompresor."
        )
        _pr_desc = (
            "Pressure ratio dalam rentang operasi normal."
            if _pr_st == "NORMAL" else
            "PR tinggi — beban kompresi meningkat. Periksa restriksi downstream."
            if _pr_st == "TINGGI" else
            "PR rendah — periksa valve bypass, kondisi seal, atau kebocoran."
        )
        _dis_desc = (
            "Tekanan discharge dalam rentang normal operasi."
            if _dis_st == "NORMAL" else
            f"Discharge pressure tinggi ({_dis_p:.2f} kg/cm2, +{_dis_dev:.2f} dari mean)."
            if _dis_st == "TINGGI" else
            f"Discharge pressure rendah ({_dis_p:.2f} kg/cm2, {_dis_dev:.2f} dari mean)."
        )
        _sp_desc = (
            f"Shaft power {_sp:.0f} kW ({_sp_load_pct:.1f}% rated) dalam batas aman."
            if _sp_st == "NORMAL" else
            f"Daya mendekati P99 ({_sp_p99_a:.0f} kW). Monitor tren."
            if _sp_st == "TINGGI" else
            f"Daya mendekati rated {rated_power:.0f} kW. Kurangi beban."
            if _sp_st == "MENDEKATI RATED" else
            f"KRITIS: Daya {_sp:.0f} kW melebihi rated! Kurangi beban segera."
        )

        _a1, _a2, _a3, _a4 = st.columns(4)
        for _col, _num, _title, _val, _unit, _st, _cl, _desc, _ref in [
            (_a1,"①","Polytropic Head",   f"{_ph:.2f}","kJ/kg",_ph_st,_ph_cl,_ph_desc,
             f"P25-P75: {_ph_p25:.2f}–{_ph_p75:.2f} | Mean: {_ph_mean:.2f}"),
            (_a2,"②","Pressure Ratio",    f"{_pr:.4f}","",_pr_st,_pr_cl,_pr_desc,
             f"P25-P75: {_pr_p25_a:.3f}–{_pr_p75_a:.3f} | Mean: {_pr_mean_a:.4f}"),
            (_a3,"③","Discharge Pressure",f"{_dis_p:.2f}","kg/cm2",_dis_st,_dis_cl,_dis_desc,
             f"Mean ref: {_dis_mean:.2f} | Design: {_design_dis}"),
            (_a4,"④","Shaft Power",       f"{_sp:.0f}","kW",_sp_st,_sp_cl,_sp_desc,
             f"Mean: {_sp_mean_a:.0f} | P99: {_sp_p99_a:.0f} | Rated: {rated_power:.0f}"),
        ]:
            with _col:
                st.markdown(f"""
<div style='background:#1a1f2e;border-radius:8px;padding:14px 12px;
            border-top:4px solid {_cl};min-height:260px'>
  <div style='font-size:12px;color:#aaa;margin-bottom:4px'>{_num} {_title}</div>
  <div style='font-size:24px;font-weight:bold;color:{_cl}'>{_val} <span style='font-size:14px'>{_unit}</span></div>
  <div style='font-size:11px;background:{_cl}22;color:{_cl};padding:2px 8px;
              border-radius:10px;display:inline-block;margin:4px 0'>{_st}</div>
  <hr style='border-color:#2a2f3e;margin:8px 0'>
  <div style='font-size:11px;color:#ccc;line-height:1.55'>{_desc}</div>
  <div style='font-size:10px;color:#666;margin-top:8px'>{_ref}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Kesimpulan Akhir Operasi C-1001B")

        _zone_val     = perf_status.zone.value if perf_status else "Tidak Diketahui"
        _surge_margin = perf_status.surge_margin_pct if perf_status else 0
        _crit_count   = sum([_ph_st=="KRITIS", _sp_st=="OVERLOAD",
                             "Z1" in _zone_val, "Z2" in _zone_val])
        _warn_count   = sum([_ph_st in ("TINGGI","RENDAH"), _pr_st in ("TINGGI","RENDAH"),
                             _dis_st in ("TINGGI","RENDAH"), _sp_st in ("TINGGI","MENDEKATI RATED")])

        if _crit_count > 0:   _ov_color="#e74c3c"; _ov_label="INTERVENSI SEGERA"; _ov_badge="KRITIS"
        elif _warn_count >= 2:_ov_color="#f39c12"; _ov_label="PERLU PERHATIAN";   _ov_badge="PERINGATAN"
        elif _warn_count == 1:_ov_color="#e67e22"; _ov_label="MONITOR LEBIH KETAT";_ov_badge="WASPADA"
        else:                 _ov_color="#2ecc71"; _ov_label="OPERASI NORMAL";    _ov_badge="AMAN"

        _concl = [
            f"Kompresor C-1001B beroperasi pada <b>{_zone_val}</b> dengan surge margin <b>{_surge_margin:.1f}%</b>.",
            f"<b>①</b> Polytropic head {_ph:.2f} kJ/kg: {_ph_desc}",
            f"<b>②</b> Pressure ratio {_pr:.4f}: {_pr_desc}",
            f"<b>③</b> Discharge pressure {_dis_p:.2f} kg/cm2: {_dis_desc}",
            f"<b>④</b> Shaft power {_sp:.0f} kW ({_sp_load_pct:.1f}% rated): {_sp_desc}",
        ]
        _rec = (perf_status.alert_messages[0] if perf_status and perf_status.alert_messages
                else "Lanjutkan pemantauan rutin.")

        st.markdown(f"""
<div style='background:#1a1f2e;border-radius:10px;padding:20px 22px;
            border-left:5px solid {_ov_color};margin-top:4px'>
  <div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>
    <span style='background:{_ov_color};color:#000;font-weight:bold;font-size:13px;
                 padding:4px 14px;border-radius:20px'>{_ov_badge}</span>
    <span style='font-size:17px;font-weight:bold;color:{_ov_color}'>{_ov_label}</span>
  </div>
  {''.join([f"<p style='margin:6px 0;font-size:13px;color:#ddd;line-height:1.5'>• {p}</p>" for p in _concl])}
  <div style='background:#111827;border-radius:6px;padding:10px 14px;margin-top:12px'>
    <span style='font-size:12px;color:#aaa'><b>Rekomendasi:</b> {_rec}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Performance table
    if perf_status and is_pump:
        st.markdown("**Ringkasan Performa Pompa**")
        st.dataframe(pd.DataFrame({
            "Parameter": ["Equipment ID","Flow Rate","Head","Power","NPSHa","Efisiensi","Zona Operasi"],
            "Nilai": [
                perf_status.equipment_id,
                f"{perf_status.flow:.1f} GPM",
                f"{perf_status.head:.1f} ft",
                f"{perf_status.power:.1f} hp",
                f"{perf_status.npsha:.1f} ft",
                f"{perf_status.efficiency:.2f} %",
                perf_status.zone.value,
            ]
        }), use_container_width=True, hide_index=True)

    elif perf_status and not is_pump:
        st.markdown("**Ringkasan Performa Kompresor C-1001B**")
        st.dataframe(pd.DataFrame({
            "Parameter": [
                "Equipment ID","Inlet Flow","Pressure Ratio","Polytropic Head",
                "Shaft Power","Poly Efficiency","Speed","Surge Margin","Zona Operasi"
            ],
            "Nilai": [
                perf_status.equipment_id,
                f"{perf_status.inlet_flow:.2f} MMSCFD",
                f"{perf_status.pressure_ratio:.4f}",
                f"{perf_status.polytropic_head:.2f} kJ/kg",
                f"{perf_status.shaft_power:.1f} kW",
                f"{perf_status.poly_efficiency:.2f} %",
                f"{perf_status.speed_rpm:.0f} RPM",
                f"{perf_status.surge_margin_pct:.1f} %",
                perf_status.zone.value,
            ]
        }), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
# TAB 3 — ML ANOMALY (tidak berubah)
# ════════════════════════════════════════════════

with tab3:
    st.subheader("🤖 Deteksi Anomali — BiLSTM Autoencoder + Rule Engine")

    col_ml1, col_ml2, col_ml3 = st.columns(3)
    col_ml1.metric("Total Data Poin", f"{len(df):,}")
    col_ml2.metric("Anomali Terdeteksi", f"{n_anomaly:,}", delta=f"{anomaly_pct:.1f}%",
                   delta_color="inverse")
    col_ml3.metric("MAE Threshold", f"{threshold:.4f}")

    st.markdown("---")

    st.subheader("🗓️ Heatmap Anomali")
    df_h        = df.copy()
    df_h["hour"]= pd.to_datetime(df_h["timestamp"]).dt.hour
    df_h["day"] = pd.to_datetime(df_h["timestamp"]).dt.strftime("%Y-%m-%d")
    if len(df_h["day"].unique()) > 1:
        pivot = df_h.pivot_table(index="hour", columns="day", values="mae", aggfunc="mean")
        fig_hmap = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                             labels={"color": "MAE"}, title="Rata-rata MAE per Jam per Hari")
        fig_hmap.update_layout(template="plotly_dark", height=300,
                               margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hmap, use_container_width=True)
    else:
        st.info("Perlu data minimal 2 hari untuk menampilkan heatmap.")

    st.subheader("📊 Fitur saat Anomali vs Normal")
    if len(df[df["is_anomaly"]]) > 0 and avail_features:
        df_normal_mean = df[~df["is_anomaly"]][avail_features].mean()
        df_anom_mean   = df[df["is_anomaly"]][avail_features].mean()
        diff_pct = ((df_anom_mean - df_normal_mean) / df_normal_mean.abs() * 100).fillna(0)

        fig_bar = go.Figure(go.Bar(
            x=avail_features, y=diff_pct.values,
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in diff_pct.values],
            text=[f"{v:+.1f}%" for v in diff_pct.values], textposition="outside"
        ))
        fig_bar.update_layout(template="plotly_dark", height=300,
                              title="Deviasi Fitur Saat Anomali vs Normal (%)",
                              xaxis_title="Fitur", yaxis_title="Deviasi (%)",
                              margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Tidak ada anomali terdeteksi dalam rentang data ini.")

    if n_anomaly > 0:
        st.subheader("📋 Rekomendasi Tindakan")
        eq_type = "pump" if is_pump else "compressor"
        anomaly_class = perf_status.zone.value if (not is_pump and perf_status) else "Equipment"
        action = get_recommended_action(anomaly_class, eq_type)
        st.warning(f"**Zona Operasi:** {anomaly_class}\n\n**Rekomendasi:** {action}")

    _date_label = (f"{st.session_state['_g_start'].strftime('%d %b %Y')} — "
                   f"{st.session_state['_g_end'].strftime('%d %b %Y')}")
    _n_anom_disp = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    st.markdown("---")
    st.markdown(f"#### 🔎 Data Mentah — Rentang {_date_label}")
    st.info(
        f"**{len(df):,}** baris data dalam rentang ini. "
        f"Anomali terdeteksi: **{_n_anom_disp}** baris "
        f"({_n_anom_disp / max(len(df), 1) * 100:.1f}%).",
        icon="📅"
    )

    with st.expander(f"Tampilkan tabel data ({len(df):,} baris)", expanded=True):
        _disp_cols = ["timestamp"] + [c for c in avail_features if c in df.columns]
        if "mae" in df.columns:        _disp_cols += ["mae"]
        if "is_anomaly" in df.columns: _disp_cols += ["is_anomaly"]
        _df_disp = df[_disp_cols].sort_values("timestamp", ascending=False).copy()

        _anom_flags = (
            _df_disp["is_anomaly"].astype(bool)
            if "is_anomaly" in _df_disp.columns
            else pd.Series([False] * len(_df_disp), index=_df_disp.index)
        )
        _anom_vals = _anom_flags.values

        # Kolom Status selalu ada sebagai indikator visual teks
        _df_disp.insert(1, "Status", _anom_flags.map({True: "⚠️ ANOMALI", False: "✅ Normal"}))

        _col_cfg = {
            "Status": st.column_config.TextColumn("Status", width="small"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD/MM/YY HH:mm"),
            "mae": st.column_config.NumberColumn("MAE", format="%.5f"),
            "is_anomaly": st.column_config.CheckboxColumn("Anomali"),
        }

        # Streamlit strips Styler bila total cells > ~150k
        # Gunakan batas 12.000 baris agar highlight baris merah tetap aktif
        _MAX_STYLED_ROWS = 12_000

        def _hl_col(col):
            return [
                "background-color: #4a0000; color: #ff6b6b" if f else ""
                for f in _anom_vals[: len(col)]
            ]

        if len(_df_disp) <= _MAX_STYLED_ROWS:
            pd.set_option("styler.render.max_elements",
                          _df_disp.shape[0] * _df_disp.shape[1] + 1)
            st.dataframe(
                _df_disp.style.apply(_hl_col, axis=0),
                use_container_width=True, hide_index=True,
                column_config=_col_cfg,
            )
        else:
            # Dataset terlalu besar untuk Streamlit Styler → tampilkan semua tanpa warna baris
            # + tabel terpisah anomali-saja yang selalu berwarna merah
            st.caption(
                f"⚠️ {len(_df_disp):,} baris terlalu besar untuk highlight penuh. "
                "Tabel lengkap ditampilkan tanpa warna baris; "
                "highlight merah tersedia di bagian **'Baris Anomali'** di bawah. "
                "Persempit rentang tanggal untuk highlight di seluruh tabel."
            )
            st.dataframe(
                _df_disp,
                use_container_width=True, hide_index=True,
                column_config=_col_cfg,
            )

            # Tabel anomali-saja: selalu kecil, selalu merah
            _df_anom_only = _df_disp[_anom_vals].copy()
            if len(_df_anom_only) > 0:
                st.markdown(
                    f"<div style='margin-top:12px; padding:6px 12px; "
                    f"background:#4a0000; border-radius:6px; color:#ff6b6b; font-weight:bold;'>"
                    f"🔴 Baris Anomali Terdeteksi ({len(_df_anom_only):,} baris)</div>",
                    unsafe_allow_html=True
                )
                _anom_only_flags = pd.Series([True] * len(_df_anom_only))
                def _hl_col_anom(col):
                    return ["background-color: #4a0000; color: #ff6b6b"] * len(col)
                pd.set_option("styler.render.max_elements",
                              _df_anom_only.shape[0] * _df_anom_only.shape[1] + 1)
                st.dataframe(
                    _df_anom_only.style.apply(_hl_col_anom, axis=0),
                    use_container_width=True, hide_index=True,
                    column_config=_col_cfg,
                )
            else:
                st.success("Tidak ada anomali dalam rentang tanggal ini.")


# ════════════════════════════════════════════════
# TAB 4 — LOG & HISTORY (tidak berubah)
# ════════════════════════════════════════════════

with tab4:
    st.subheader("📋 Log Anomali Database")

    try:
        df_log = get_anomaly_log(limit=100)
        if len(df_log) == 0:
            st.info("Belum ada log anomali tersimpan.")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_eq = st.multiselect("Filter Equipment",
                    df_log["equipment_id"].unique().tolist(),
                    default=df_log["equipment_id"].unique().tolist())
            with col_f2:
                filter_status = st.multiselect("Filter Status",
                    df_log["status"].dropna().unique().tolist(),
                    default=df_log["status"].dropna().unique().tolist())

            mask = df_log["equipment_id"].isin(filter_eq)
            if filter_status:
                mask &= df_log["status"].isin(filter_status)
            df_filtered = df_log[mask]
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)
            st.caption(f"{len(df_filtered)} entri dari {len(df_log)} total log")
    except Exception as e:
        st.error(f"Gagal membaca log: {e}")

    st.markdown("---")
    st.subheader("📦 Statistik Historis")

    if avail_features:
        st.markdown("**Statistik Deskriptif — Data Saat Ini**")
        st.dataframe(df[avail_features + ["mae"]].describe().round(3),
                     use_container_width=True)

    if avail_features and len(avail_features) >= 2:
        with st.expander("🔀 Scatter Matrix Fitur"):
            fig_scatter = px.scatter_matrix(
                df, dimensions=avail_features[:4],
                color="is_anomaly",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                labels={"is_anomaly": "Anomali"},
                title="Scatter Matrix — Normal vs Anomali"
            )
            fig_scatter.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("💾 Ekspor Data")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Data (CSV)", data=csv_out,
            file_name=f"integrate_{eq_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv")
    with col_e2:
        df_anom_export = df[df["is_anomaly"]].copy()
        if len(df_anom_export) > 0:
            csv_anom = df_anom_export.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Log Anomali (CSV)", data=csv_anom,
                file_name=f"anomali_{eq_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        else:
            st.info("Tidak ada anomali untuk diekspor.")


# ════════════════════════════════════════════════
# TAB 5 — IMPLEMENTASI (tidak berubah, sama persis)
# ════════════════════════════════════════════════

with tab5:
    _eq_label = "Pompa P-1001A (DMF Pump)" if is_pump else "Kompresor C-1001B (BCL305)"
    st.subheader(f"🏭 Implementasi Lapangan — {_eq_label}")
    st.markdown(
        f"Panduan deployment **INTEGRATE** untuk **{_eq_label}** dari lingkungan development "
        "ke sistem lapangan industri migas yang terintegrasi dengan DCS, sensor real-time, dan jaringan industrial."
    )

    # ── SECTION 1: Architecture Diagram 5 Layer ──────────────────────
    st.markdown("---")
    st.markdown("### 📐 Arsitektur Sistem 5 Layer")
    st.caption("Dari sensor fisik di lapangan hingga dashboard operator & monitoring jarak jauh")

    _fig_arch = go.Figure()
    _fig_arch.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 14]),
        height=600, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )

    _arch_layers = [
        dict(y0=0.2, y1=1.6, fill="#0d2b1a", line_c="#27ae60",
             title="LAYER 1 — ROTATING EQUIPMENT (Physical Asset)",
             sub="Compressor C-1001B BCL305 | Pump P-1001A | Anti-Surge Valve | Piping"),
        dict(y0=2.2, y1=3.6, fill="#0d1b2e", line_c="#2980b9",
             title="LAYER 2 — FIELD INSTRUMENTS & SENSORS",
             sub="Pressure Transmitter | Coriolis Flow Meter | RTD Temperature | Vibration Probe | Speed Sensor"),
        dict(y0=4.2, y1=5.6, fill="#1e0d2e", line_c="#8e44ad",
             title="LAYER 3 — DCS / HISTORIAN (Exaquantum)",
             sub="Yokogawa CS3000 DCS | Exaquantum R3.80 Historian | OPC-UA Server"),
        dict(y0=6.2, y1=7.6, fill="#0d2e0d", line_c="#16a085",
             title="LAYER 4 — INTEGRATE EDGE SERVER (Control Room)",
             sub="Industrial PC 24/7 | Python 3.11 | Streamlit | BiLSTM ML Engine | Rule Engine | SQLite | Email Alert"),
        dict(y0=8.2, y1=9.6, fill="#2e0d0d", line_c="#c0392b",
             title="LAYER 5 — CONTROL ROOM & REMOTE MONITORING",
             sub="Operator Workstation | Head Office VPN Tunnel | SMS / Email Notifikasi Engineer"),
    ]

    for _ly in _arch_layers:
        _ym = (_ly["y0"] + _ly["y1"]) / 2
        _fig_arch.add_shape(type="rect", x0=0.3, y0=_ly["y0"], x1=9.7, y1=_ly["y1"],
                            fillcolor=_ly["fill"], line=dict(color=_ly["line_c"], width=2))
        _fig_arch.add_annotation(x=5, y=_ym + 0.30,
                                 text=f"<b style='font-size:13px'>{_ly['title']}</b>",
                                 font=dict(color="white", size=12), showarrow=False)
        _fig_arch.add_annotation(x=5, y=_ym - 0.25, text=_ly["sub"],
                                 font=dict(color="#aaaaaa", size=10), showarrow=False)

    _conn_arrows = [
        (1.6, 2.2, "4-20mA / HART / PROFIBUS"),
        (3.6, 4.2, "OPC-DA / Modbus TCP"),
        (5.6, 6.2, "OPC-UA Client / Direct SQL Query"),
        (7.6, 8.2, "HTTP (LAN) / VPN Tunnel"),
    ]
    for _ya, _yb, _lbl in _conn_arrows:
        _ym = (_ya + _yb) / 2
        _fig_arch.add_annotation(x=5, y=_yb, ax=5, ay=_ya, axref="x", ayref="y",
                                 arrowhead=3, arrowsize=1.2, arrowwidth=2,
                                 arrowcolor="#f39c12", showarrow=True, text="")
        _fig_arch.add_annotation(x=7.5, y=_ym, text=f"<i>{_lbl}</i>",
                                 font=dict(color="#f39c12", size=9), showarrow=False,
                                 bgcolor="#1a1408", bordercolor="#f39c12",
                                 borderwidth=1, borderpad=3)

    _fig_arch.add_annotation(x=5, y=13.2,
                             text="<b>INTEGRATE — Field Deployment Architecture (API 617 / API 610)</b>",
                             font=dict(color="#00b4d8", size=15), showarrow=False)
    st.plotly_chart(_fig_arch, use_container_width=True)

    # ── SECTION 2: Hardware & Software Specs ─────────────────────────
    st.markdown("---")
    st.markdown("### 🔧 Bill of Materials — Hardware & Software")

    _col_hw1, _col_hw2 = st.columns(2)

    with _col_hw1:
        st.markdown("**Hardware Utama**")
        st.dataframe(pd.DataFrame({
            "Komponen": [
                "Edge Server (Industrial PC)",
                "Managed Industrial Switch",
                "Industrial Firewall",
                "UPS Backup Power",
                "Operator Workstation",
                "OPC-UA Gateway",
                "Serial-to-Ethernet Converter",
            ],
            "Model / Spesifikasi": [
                "ADVANTECH MIC-7700 | i7 | 32GB RAM | 1TB SSD",
                "Cisco IE-2000-16TC | 24-port | DIN-rail",
                "Hirschmann EAGLE One | Industrial-grade",
                "APC Smart-UPS 1500VA | 230V | 30 min backup",
                "Dell OptiPlex 7090 | i5 | 16GB | 27 inci",
                "Kepware KEPServerEX v6.15",
                "Moxa NPort 5150 | RS-232/485 ke Ethernet",
            ],
            "Qty": [1, 1, 1, 2, 2, 1, 2],
            "Fungsi Utama": [
                "Jalankan INTEGRATE (ML + Dashboard) 24/7",
                "Jaringan industrial LAN terisolasi",
                "Segmentasi & keamanan jaringan OT/IT",
                "Power backup server + switch jika PLN mati",
                "Akses dashboard oleh operator",
                "Bridge OPC-UA ke DCS Yokogawa",
                "Koneksi instrument legacy Modbus RTU",
            ]
        }), use_container_width=True, hide_index=True)

    with _col_hw2:
        st.markdown("**Software Stack (Semua Open Source / Free)**")
        st.dataframe(pd.DataFrame({
            "Software": [
                "Ubuntu Server 22.04 LTS", "Python 3.11.x", "Streamlit 1.35.0",
                "TensorFlow 2.15.0", "opcua-asyncio", "SQLite + SQLAlchemy",
                "Nginx (reverse proxy)", "Supervisor",
            ],
            "Fungsi": [
                "OS stabil & aman untuk server industri",
                "Runtime seluruh sistem INTEGRATE",
                "Web dashboard engine (no browser plugin)",
                "BiLSTM Autoencoder inference",
                "Koneksi real-time ke OPC-UA Server DCS",
                "Penyimpanan log anomali lokal",
                "HTTPS + akses multi-user LAN",
                "Auto-restart INTEGRATE jika crash",
            ],
            "Lisensi": [
                "Free (Open Source)", "Free", "Apache 2.0", "Apache 2.0",
                "MIT", "MIT/BSD", "BSD", "MIT"
            ]
        }), use_container_width=True, hide_index=True)

    st.markdown("**Perbandingan: Development (Laptop) vs Production (Field Server)**")
    st.dataframe(pd.DataFrame({
        "Aspek": [
            "Hardware", "OS", "Sumber Data", "Protokol", "Reliabilitas",
            "Keamanan Jaringan", "Kapasitas Historis", "Akses Multi-User", "Pemulihan Otomatis"
        ],
        "Laptop (Development Sekarang)": [
            "Consumer laptop", "Windows 10/11", "File Excel / CSV manual upload",
            "File system lokal", "Hanya aktif saat user membuka",
            "Tidak ada segmentasi jaringan", "Terbatas kapasitas RAM/disk",
            "Tidak (local only)", "Restart manual oleh user"
        ],
        "Industrial PC (Production Lapangan)": [
            "ADVANTECH MIC-7700 rated 24/7", "Ubuntu Server 22.04 LTS",
            "OPC-UA polling otomatis dari Exaquantum",
            "OPC-UA / Modbus TCP / Direct SQL",
            "99.9% uptime, auto-restart via Supervisor",
            "Firewall industrial + VLAN OT terisolasi",
            "1TB SSD lokal + archive NAS opsional",
            "Ya (LAN) + VPN remote head office",
            "Supervisor watchdog + UPS power backup"
        ]
    }), use_container_width=True, hide_index=True)

    # ── SECTION 3: OPC-UA Tag Mapping ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔗 Pemetaan Tag OPC-UA — Exaquantum ke INTEGRATE")
    st.caption("Tag NodeId yang harus dikonfigurasi di KEPServerEX untuk equipment aktif")

    if is_pump:
        st.dataframe(pd.DataFrame({
            "Tag Exaquantum (OPC-UA NodeId)": [
                "ns=2;s=P1001A.FlowRate.PV",
                "ns=2;s=P1001A.SuctionPressure.PV",
                "ns=2;s=P1001A.DischargePressure.PV",
                "ns=2;s=P1001A.Temperature.PV",
                "ns=2;s=P1001A.MotorCurrent.PV",
                "ns=2;s=P1001A.SealPressure.PV",
            ],
            "Kolom INTEGRATE": [
                "flow_rate", "suction_pressure", "discharge_pressure",
                "temperature", "motor_current", "seal_pressure",
            ],
            "Satuan": ["BPD", "psig", "psig", "degF", "A", "psig"],
            "Interval Poll": ["5 min", "5 min", "5 min", "5 min", "5 min", "5 min"],
            "Alert Threshold": [
                "< 5000 BPD (AOR Low)",
                "< 140 atau > 185 psig",
                "< 900 atau > 1200 psig",
                "> 130 degF",
                "> 200 A",
                "< 25 psig (kebocoran seal)",
            ],
            "Sensor Fisik": [
                "FT-201 Coriolis Flow Meter (Emerson F300)",
                "PT-201 Pressure Transmitter (Rosemount 3051)",
                "PT-202 Pressure Transmitter (Rosemount 3051)",
                "TT-201 RTD Pt100 (4-wire)",
                "IT-201 Current Transformer (100/5A)",
                "PT-203 Differential Pressure Tx (Rosemount 2051)",
            ],
        }), use_container_width=True, hide_index=True)
    else:
        st.dataframe(pd.DataFrame({
            "Tag Exaquantum (OPC-UA NodeId)": [
                "ns=2;s=C1001B.InletFlow.PV",
                "ns=2;s=C1001B.SuctionPressure.PV",
                "ns=2;s=C1001B.DischargePressure.PV",
                "ns=2;s=C1001B.SuctionTemp.PV",
                "ns=2;s=C1001B.DischargeTemp.PV",
                "ns=2;s=C1001B.ShaftPower.PV",
                "ns=2;s=C1001B.Speed.PV",
                "ns=2;s=C1001B.PressureRatio.PV",
            ],
            "Kolom INTEGRATE": [
                "inlet_flow", "suction_pressure", "discharge_pressure",
                "T_suction_F", "T_discharge_F", "shaft_power", "speed_rpm", "pressure_ratio",
            ],
            "Satuan": ["MMSCFD", "kg/cm2", "kg/cm2", "degF", "degF", "kW", "RPM", "-"],
            "Interval Poll": ["1 min", "1 min", "1 min", "5 min", "5 min", "1 min", "1 min", "1 min"],
            "Alert Threshold": [
                "< 27.09 MMSCFD = SURGE",
                "< 33 atau > 45 kg/cm2",
                "< 55 atau > 70 kg/cm2",
                "> 115 degF",
                "> 220 degF",
                "> 1408 kW (110% rated)",
                "< 8500 atau > 13500 RPM",
                "< 1.2 atau > 2.0",
            ],
            "Sensor Fisik": [
                "FT-101 Coriolis Flow Meter (Emerson F300)",
                "PT-101 Pressure Transmitter (Rosemount 3051)",
                "PT-102 Pressure Transmitter (Rosemount 3051)",
                "TT-101 RTD Pt100 (suction)",
                "TT-102 RTD Pt100 (discharge)",
                "Power Meter via Speed × Torque",
                "ST-101 Magnetic Speed Sensor",
                "Calculated: PT-102 / PT-101",
            ],
        }), use_container_width=True, hide_index=True)

    # ── SECTION 4: Field Scenario Simulation ─────────────────────────
    st.markdown("---")

    def _sim_fig(title, ylab):
        _f = go.Figure()
        _f.update_layout(
            title=title, height=300,
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
            font=dict(color="white"), margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor="#2a2f3e"),
            yaxis=dict(gridcolor="#2a2f3e", title=ylab),
            legend=dict(bgcolor="#1a1f2e", bordercolor="#444")
        )
        return _f

    np.random.seed(99)
    _n_sim = 60
    _t_sim = pd.date_range("2025-08-27 08:00", periods=_n_sim, freq="1min")
    _idx = np.arange(_n_sim)

    if is_pump:
        _t_cav_start = 38
        st.markdown("### 🔄 Simulasi Skenario Lapangan — Kavitasi / Low-Flow Pompa P-1001A")
        st.caption(
            "Simulasi interaktif bagaimana INTEGRATE mendeteksi dan merespons kejadian kavitasi "
            "akibat low-flow pada pompa P-1001A: dari sensor → DCS → OPC-UA → ML Engine → Alert operator"
        )

        # Synthetic cavitation scenario
        _flow_p = np.clip(np.where(
            _idx < _t_cav_start,
            12720 + np.random.normal(0, 150, _n_sim),
            12720 - np.linspace(0, 9500, _n_sim) + np.random.normal(0, 80, _n_sim)
        ), 1200, 14000)
        _dp_p = np.clip(np.where(
            _idx < _t_cav_start,
            918 + np.random.normal(0, 5, _n_sim),
            918 - np.linspace(0, 310, _n_sim) + np.random.normal(0, 4, _n_sim)
        ), 300, 960)
        _npsha_p = np.clip(np.where(
            _idx < _t_cav_start,
            28 + np.random.normal(0, 0.4, _n_sim),
            28 - np.linspace(0, 18, _n_sim) + np.random.normal(0, 0.3, _n_sim)
        ), 4, 32)
        _curr_p = np.clip(np.where(
            _idx < _t_cav_start,
            145 + np.random.normal(0, 2, _n_sim),
            145 + np.linspace(0, 55, _n_sim) + np.random.normal(0, 2, _n_sim)
        ), 140, 210)

        _pc1, _pc2 = st.columns(2)
        with _pc1:
            _pf1 = _sim_fig("Flow Rate — Low-Flow Menuju Kavitasi", "BPD")
            _pf1.add_trace(go.Scatter(x=_t_sim, y=_flow_p, name="Flow Rate",
                                      line=dict(color="#3498db", width=2)))
            _pf1.add_hline(y=5700, line=dict(color="#e74c3c", width=2, dash="dash"),
                           annotation_text="AOR Min 5700 BPD", annotation_font_color="#e74c3c",
                           annotation_position="top left")
            _pf1.add_hline(y=7632, line=dict(color="#f39c12", width=1.5, dash="dot"),
                           annotation_text="POR Min 7632 BPD", annotation_font_color="#f39c12",
                           annotation_position="bottom right")
            _pf1.add_vrect(x0=str(_t_sim[_t_cav_start]), x1=str(_t_sim[-1]),
                           fillcolor="#e74c3c", opacity=0.12, line_width=0,
                           annotation_text="KAVITASI!", annotation_font_color="#e74c3c",
                           annotation_position="top left")
            st.plotly_chart(_pf1, use_container_width=True)

            _pf3 = _sim_fig("Motor Current — Spike saat Kavitasi", "Ampere")
            _pf3.add_trace(go.Scatter(x=_t_sim, y=_curr_p, name="Motor Current",
                                      line=dict(color="#9b59b6", width=2)))
            _pf3.add_hline(y=175, line=dict(color="#f39c12", width=1.5, dash="dash"),
                           annotation_text="Warning 175 A", annotation_font_color="#f39c12")
            _pf3.add_hline(y=200, line=dict(color="#e74c3c", width=2, dash="dash"),
                           annotation_text="KRITIS 200 A", annotation_font_color="#e74c3c")
            _pf3.add_vrect(x0=str(_t_sim[_t_cav_start]), x1=str(_t_sim[-1]),
                           fillcolor="#e74c3c", opacity=0.12, line_width=0)
            st.plotly_chart(_pf3, use_container_width=True)

        with _pc2:
            _pf2 = _sim_fig("Differential Pressure — Turun saat Low-Flow", "psig")
            _pf2.add_trace(go.Scatter(x=_t_sim, y=_dp_p, name="Diff Pressure",
                                      line=dict(color="#2ecc71", width=2)))
            _pf2.add_vrect(x0=str(_t_sim[_t_cav_start]), x1=str(_t_sim[-1]),
                           fillcolor="#e74c3c", opacity=0.12, line_width=0,
                           annotation_text="KAVITASI!", annotation_font_color="#e74c3c",
                           annotation_position="top left")
            st.plotly_chart(_pf2, use_container_width=True)

            _npshr_vals = 3.0 + 0.0001 * (_flow_p * 42.0 / 1440.0) ** 2
            _pf4 = _sim_fig("NPSHa vs NPSHr — Margin Kavitasi", "ft")
            _pf4.add_trace(go.Scatter(x=_t_sim, y=_npsha_p, name="NPSHa",
                                      line=dict(color="#3498db", width=2)))
            _pf4.add_trace(go.Scatter(x=_t_sim, y=_npshr_vals, name="NPSHr",
                                      line=dict(color="#e74c3c", width=2, dash="dash")))
            _pf4.add_vrect(x0=str(_t_sim[_t_cav_start]), x1=str(_t_sim[-1]),
                           fillcolor="#e74c3c", opacity=0.12, line_width=0,
                           annotation_text="NPSHa < NPSHr", annotation_font_color="#e74c3c",
                           annotation_position="top left")
            st.plotly_chart(_pf4, use_container_width=True)

        st.markdown("#### Alur Respons Sistem INTEGRATE saat Kavitasi Terdeteksi")
        for _color, _title, _msg in [
            ("#2980b9", "Step 1 — Sensor Lapangan",
             "Flow Rate turun dari 12720 BPD → 3220 BPD. FT-201 Coriolis & PT-201 mengirim sinyal 4-20mA ke DCS."),
            ("#8e44ad", "Step 2 — DCS Yokogawa CS3000",
             "DCS memproses sinyal, mencatat ke Exaquantum historian setiap 5 menit."),
            ("#16a085", "Step 3 — OPC-UA Polling",
             "INTEGRATE Edge Server polling OPC-UA setiap 5 menit. Data masuk ke pipeline preprocessing Python."),
            ("#d35400", "Step 4 — Rule Engine (Instant)",
             "check_pump_performance() → Zone: AOR Low! Flow = 3220 BPD < AOR Min 5700 BPD. Deteksi < 1 detik."),
            ("#c0392b", "Step 5 — BiLSTM ML Engine",
             "MAE = 0.38 >> Threshold = 0.032. Anomali KRITIS dikonfirmasi oleh AI (pola kavitasi terdeteksi)."),
            ("#e74c3c", "Step 6 — Alert Otomatis",
             "Email + SMS ke engineer: 'BAHAYA KAVITASI P-1001A | Flow=3220 BPD | Buka bypass valve / cek suction SEGERA!'"),
            ("#27ae60", "Step 7 — Dashboard Operator",
             "Dashboard merah: Zone AOR Low, NPSHa < NPSHr, rekomendasi tindakan darurat tampil."),
        ]:
            st.markdown(
                f"<div style='background:{_color}1a; border-left:4px solid {_color}; "
                f"padding:8px 14px; border-radius:5px; margin:5px 0;'>"
                f"<b style='color:{_color}'>{_title}</b><br>"
                f"<span style='color:#cccccc; font-size:13px'>{_msg}</span></div>",
                unsafe_allow_html=True
            )

    else:
        _t_surge_start = 42
        st.markdown("### 🔄 Simulasi Skenario Lapangan — Kejadian Surge Kompresor C-1001B")
        st.caption(
            "Simulasi interaktif bagaimana INTEGRATE mendeteksi dan merespons kejadian surge "
            "secara real-time: dari sensor → DCS → OPC-UA → ML Engine → Alert operator"
        )

        _flow_sim = np.clip(np.where(
            _idx < _t_surge_start,
            53.4 + np.random.normal(0, 0.8, _n_sim),
            53.4 - np.linspace(0, 32, _n_sim) + np.random.normal(0, 0.4, _n_sim)
        ), 18, 60)
        _hp_sim = np.clip(np.where(
            _idx < _t_surge_start,
            79.5 + np.random.normal(0, 0.5, _n_sim),
            79.5 + np.linspace(0, 18, _n_sim) + np.random.normal(0, 0.4, _n_sim)
        ), 60, 102)
        _sp_sim = np.clip(np.where(
            _idx < _t_surge_start,
            1243 + np.random.normal(0, 15, _n_sim),
            1243 + np.linspace(0, 320, _n_sim) + np.random.normal(0, 12, _n_sim)
        ), 800, 1600)
        _pr_sim = np.where(
            _idx < _t_surge_start,
            1.767 + np.random.normal(0, 0.01, _n_sim),
            1.767 + np.linspace(0, 0.12, _n_sim) + np.random.normal(0, 0.008, _n_sim)
        )

        _sc1, _sc2 = st.columns(2)
        with _sc1:
            _f1 = _sim_fig("Inlet Flow — Surge Event Simulation", "MMSCFD")
            _f1.add_trace(go.Scatter(x=_t_sim, y=_flow_sim, name="Inlet Flow",
                                     line=dict(color="#3498db", width=2)))
            _f1.add_hline(y=27.09, line=dict(color="#e74c3c", width=2, dash="dash"),
                          annotation_text="Surge Line 27.09", annotation_font_color="#e74c3c",
                          annotation_position="top left")
            _f1.add_hline(y=29.54, line=dict(color="#f39c12", width=1.5, dash="dot"),
                          annotation_text="Protection 29.54", annotation_font_color="#f39c12",
                          annotation_position="bottom right")
            _f1.add_vrect(x0=str(_t_sim[_t_surge_start]), x1=str(_t_sim[-1]),
                          fillcolor="#e74c3c", opacity=0.12, line_width=0,
                          annotation_text="SURGE!", annotation_font_color="#e74c3c",
                          annotation_position="top left")
            st.plotly_chart(_f1, use_container_width=True)

            _f3 = _sim_fig("Shaft Power — Overload saat Surge", "kW")
            _f3.add_trace(go.Scatter(x=_t_sim, y=_sp_sim, name="Shaft Power",
                                     line=dict(color="#9b59b6", width=2)))
            _f3.add_hline(y=1280, line=dict(color="#f39c12", width=1.5, dash="dash"),
                          annotation_text="Rated 1280 kW", annotation_font_color="#f39c12")
            _f3.add_hline(y=1408, line=dict(color="#e74c3c", width=2, dash="dash"),
                          annotation_text="KRITIS 110%", annotation_font_color="#e74c3c")
            _f3.add_vrect(x0=str(_t_sim[_t_surge_start]), x1=str(_t_sim[-1]),
                          fillcolor="#e74c3c", opacity=0.12, line_width=0)
            st.plotly_chart(_f3, use_container_width=True)

        with _sc2:
            _f2 = _sim_fig("Polytropic Head — Naik saat Surge", "kJ/kg")
            _f2.add_trace(go.Scatter(x=_t_sim, y=_hp_sim, name="Poly Head",
                                     line=dict(color="#2ecc71", width=2)))
            _f2.add_vrect(x0=str(_t_sim[_t_surge_start]), x1=str(_t_sim[-1]),
                          fillcolor="#e74c3c", opacity=0.12, line_width=0,
                          annotation_text="SURGE!", annotation_font_color="#e74c3c",
                          annotation_position="top left")
            st.plotly_chart(_f2, use_container_width=True)

            _f4 = _sim_fig("Pressure Ratio — Anomali saat Surge", "-")
            _f4.add_trace(go.Scatter(x=_t_sim, y=_pr_sim, name="Pressure Ratio",
                                     line=dict(color="#f39c12", width=2)))
            _f4.add_vrect(x0=str(_t_sim[_t_surge_start]), x1=str(_t_sim[-1]),
                          fillcolor="#e74c3c", opacity=0.12, line_width=0,
                          annotation_text="SURGE!", annotation_font_color="#e74c3c",
                          annotation_position="top left")
            st.plotly_chart(_f4, use_container_width=True)

        st.markdown("#### Alur Respons Sistem INTEGRATE saat Surge Terdeteksi")
        for _color, _title, _msg in [
            ("#2980b9", "Step 1 — Sensor Lapangan",
             "Inlet Flow turun dari 53.4 MMSCFD → 21.8 MMSCFD. Pressure Transmitter & Flow Meter mengirim sinyal 4-20mA ke DCS."),
            ("#8e44ad", "Step 2 — DCS Yokogawa CS3000",
             "DCS memproses sinyal analog, mengkonversi ke nilai engineering. Exaquantum historian mencatat setiap 1 menit."),
            ("#16a085", "Step 3 — OPC-UA Polling",
             "INTEGRATE Edge Server polling OPC-UA setiap 1 menit ke Exaquantum. Data masuk ke pipeline preprocessing Python."),
            ("#d35400", "Step 4 — Rule Engine (Instant)",
             "check_compressor_performance() → Zone: Z1 SURGE! Surge Margin = -19.6%. Deteksi < 1 detik."),
            ("#c0392b", "Step 5 — BiLSTM ML Engine",
             "MAE = 0.412 >> Threshold = 0.041. Anomali KRITIS dikonfirmasi oleh AI."),
            ("#e74c3c", "Step 6 — Alert Otomatis",
             "Email + SMS ke engineer: 'BAHAYA SURGE C-1001B | Flow=21.8 MMSCFD | Buka Anti-Surge Valve SEGERA!'"),
            ("#27ae60", "Step 7 — Dashboard Operator",
             "Dashboard merah: Zone Z1, Surge Margin -19.6%, rekomendasi tindakan darurat tampil."),
        ]:
            st.markdown(
                f"<div style='background:{_color}1a; border-left:4px solid {_color}; "
                f"padding:8px 14px; border-radius:5px; margin:5px 0;'>"
                f"<b style='color:{_color}'>{_title}</b><br>"
                f"<span style='color:#cccccc; font-size:13px'>{_msg}</span></div>",
                unsafe_allow_html=True
            )

    # ── SECTION 5: Deployment Plan / Gantt ───────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Rencana Deployment — 6 Fase Implementasi (6 Minggu)")

    _gantt_data = pd.DataFrame([
        dict(Fase="Infrastruktur", Task="Fase 1: Instalasi Hardware & Jaringan",
             Start="2025-09-01", Finish="2025-09-10",
             Detail="Pasang industrial PC, switch, firewall, UPS di control room."),
        dict(Fase="Integrasi DCS", Task="Fase 2: Konfigurasi OPC-UA & Exaquantum",
             Start="2025-09-08", Finish="2025-09-18",
             Detail="Setup KEPServerEX, mapping tag OPC-UA, verifikasi dengan engineer DCS."),
        dict(Fase="Software", Task="Fase 3: Deploy INTEGRATE + End-to-End Test",
             Start="2025-09-15", Finish="2025-09-26",
             Detail="Install Python env, deploy app, test semua tab & alert dengan data live."),
        dict(Fase="ML / AI", Task="Fase 4: Retraining ML dengan Data Lokal",
             Start="2025-09-22", Finish="2025-10-03",
             Detail="Retrain BiLSTM Autoencoder dengan minimal 30 hari data DCS aktual."),
        dict(Fase="Training", Task="Fase 5: Training Operator & Engineer",
             Start="2025-09-30", Finish="2025-10-08",
             Detail="Pelatihan penggunaan dashboard, interpretasi zona, respons alert."),
        dict(Fase="Produksi", Task="Fase 6: Go-Live & Monitoring 30 Hari",
             Start="2025-10-06", Finish="2025-11-05",
             Detail="Operasional penuh. Monitoring harian, evaluasi false positive rate."),
    ])
    _fig_gantt = px.timeline(
        _gantt_data, x_start="Start", x_end="Finish", y="Task", color="Fase",
        color_discrete_map={"Infrastruktur": "#2980b9", "Integrasi DCS": "#8e44ad",
                            "Software": "#16a085", "ML / AI": "#d35400",
                            "Training": "#f39c12", "Produksi": "#27ae60"},
        hover_data={"Detail": True, "Start": True, "Finish": True, "Fase": False},
        title="Timeline Implementasi INTEGRATE"
    )
    _fig_gantt.update_yaxes(autorange="reversed")
    _fig_gantt.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e", font=dict(color="white"),
        height=340, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="#2a2f3e", title=""),
        yaxis=dict(gridcolor="#2a2f3e", title=""),
        legend=dict(bgcolor="#1a1f2e", bordercolor="#444", title="Fase"),
        title_font=dict(color="#00b4d8", size=14)
    )
    st.plotly_chart(_fig_gantt, use_container_width=True)

    _ch1, _ch2 = st.columns(2)
    with _ch1:
        st.markdown("**Pre-Deployment Checklist**")
        for _item in [
            "Rack server di control room tersedia & berventilasi",
            "Jaringan industrial LAN terpasang & tested",
            "OPC-UA server Exaquantum aktif & accessible",
            "Tag mapping diverifikasi bersama engineer DCS",
            "Firewall rules dikonfigurasi (whitelist IP server)",
            "UPS terpasang, battery tested, runtime >= 30 menit",
            "Python 3.11 + semua dependencies terinstall",
            "INTEGRATE startup berhasil, load data dari OPC-UA",
        ]:
            st.markdown(f"- [ ] {_item}")
    with _ch2:
        st.markdown("**Post Go-Live Checklist**")
        for _item in [
            "Dashboard diakses dari semua workstation operator",
            "Email alert terkirim saat uji anomali manual",
            "ML model di-retrain dengan data DCS 30+ hari",
            "Surge detection diverifikasi oleh process engineer",
            "Backup otomatis SQLite database berjalan (cron)",
            "Uptime monitoring terpasang (Supervisor + Nagios)",
            "Dokumentasi O&M diserahkan ke tim operasi",
            "Jadwal retraining model ditetapkan (setiap 3 bulan)",
        ]:
            st.markdown(f"- [ ] {_item}")

    # ── SECTION 6: Network Topology ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Topologi Jaringan Industrial")

    _fig_net = go.Figure()
    _fig_net.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 8]),
        height=420, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Network Topology — INTEGRATE Field Deployment",
                   font=dict(color="#00b4d8", size=13))
    )
    _nodes = [
        (1.0, 1.0, "Sensor\nC-1001B", "#27ae60", 0.7),
        (1.0, 3.0, "Sensor\nP-1001A", "#27ae60", 0.7),
        (3.5, 2.0, "Yokogawa\nCS3000 DCS", "#8e44ad", 0.9),
        (5.5, 2.0, "Exaquantum\nHistorian", "#2980b9", 0.9),
        (5.5, 4.5, "Firewall\nIndustrial", "#e74c3c", 0.7),
        (7.5, 2.0, "INTEGRATE\nEdge Server", "#16a085", 1.0),
        (7.5, 4.5, "Industrial\nSwitch LAN", "#f39c12", 0.7),
        (9.5, 1.0, "Operator\nWS 1", "#00b4d8", 0.6),
        (9.5, 3.0, "Operator\nWS 2", "#00b4d8", 0.6),
        (9.5, 5.0, "VPN\nHead Office", "#9b59b6", 0.6),
    ]
    _edges = [
        (0, 2, "4-20mA"), (1, 2, "HART"), (2, 3, "OPC-DA"), (3, 4, "DMZ"),
        (4, 6, "Firewall"), (3, 5, "OPC-UA"), (5, 6, "Eth"),
        (6, 7, "HTTP"), (6, 8, "HTTP"), (6, 9, "VPN"),
    ]
    for (_x1, _y1, _, _c1, _), (_x2, _y2, _, _c2, _), _lbl in [
        (_nodes[_e[0]], _nodes[_e[1]], _e[2]) for _e in _edges
    ]:
        _xm, _ym = (_x1 + _x2) / 2, (_y1 + _y2) / 2
        _fig_net.add_shape(type="line", x0=_x1, y0=_y1, x1=_x2, y1=_y2,
                           line=dict(color="#444", width=1.5, dash="dot"))
        _fig_net.add_annotation(x=_xm, y=_ym, text=f"<i>{_lbl}</i>",
                                font=dict(color="#888", size=8), showarrow=False)

    def _hex_to_rgba(h, a=0.2):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    for _nx, _ny, _lbl, _color, _r in _nodes:
        _fig_net.add_shape(type="circle",
                           x0=_nx - _r, y0=_ny - _r * 0.6,
                           x1=_nx + _r, y1=_ny + _r * 0.6,
                           fillcolor=_hex_to_rgba(_color, 0.2),
                           line=dict(color=_color, width=2))
        _fig_net.add_annotation(x=_nx, y=_ny, text=f"<b>{_lbl}</b>",
                                font=dict(color="white", size=9), showarrow=False)
    st.plotly_chart(_fig_net, use_container_width=True)

    st.success(
        "**INTEGRATE siap deploy ke lapangan tanpa biaya lisensi software tambahan. "
        "Seluruh stack menggunakan open-source. Estimasi total biaya hardware: "
        "Rp 180 juta – Rp 250 juta (termasuk industrial PC, switch, firewall, UPS, workstation).**"
    )

    # ── SECTION 7: 3D Hardware Mockup ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖼️ Ilustrasi 3D — Mock-up Instalasi Hardware Lapangan")
    st.caption("Ilustrasi interaktif — klik & drag untuk merotasi, scroll untuk zoom, klik legend untuk sembunyikan/tampilkan item.")

    def _box3d(x0, y0, z0, x1, y1, z1, color, name, opacity=0.80):
        vx = [x0, x1, x1, x0, x0, x1, x1, x0]
        vy = [y0, y0, y1, y1, y0, y0, y1, y1]
        vz = [z0, z0, z0, z0, z1, z1, z1, z1]
        ii = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
        jj = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
        kk = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
        return go.Mesh3d(x=vx, y=vy, z=vz, i=ii, j=jj, k=kk,
                         color=color, opacity=opacity, name=name,
                         showlegend=True, flatshading=True, hoverinfo="name")

    def _line3d(xs, ys, zs, color, name, width=4):
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                            line=dict(color=color, width=width),
                            name=name, showlegend=True)

    def _label3d(xs, ys, zs, texts, color="white", size=11):
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="text",
                            text=texts, textfont=dict(color=color, size=size),
                            showlegend=False, hoverinfo="skip")

    _dark_scene = dict(
        xaxis=dict(showgrid=False, showticklabels=False, title="",
                   backgroundcolor="#0e1117", showbackground=True, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, title="",
                   backgroundcolor="#0e1117", showbackground=True, zeroline=False),
        zaxis=dict(showgrid=False, showticklabels=False, title="",
                   backgroundcolor="#0e1117", showbackground=True, zeroline=False),
        bgcolor="#0e1117",
    )

    with st.expander("🏢  Ilustrasi 3D — Ruang Kontrol & Server Rack", expanded=True):
        _cr = go.Figure()
        _cr.add_trace(_box3d(0, 0, -0.06, 11, 7, 0, "#1a1a2e", "Lantai Ruang Kontrol", 0.45))
        _cr.add_trace(_box3d(0, 6.9, 0, 11, 7.0, 3.0, "#12122a", "Dinding Belakang", 0.30))
        _cr.add_trace(_box3d(-0.1, 0, 0, 0.0, 7, 3.0, "#12122a", "Dinding Kiri", 0.30))
        _cr.add_trace(_box3d(0.3, 0.3, 0, 1.2, 1.1, 2.1, "#2c3e50", "Lemari Rack Server", 0.90))
        _cr.add_trace(_box3d(0.35, 0.32, 0.08, 1.15, 1.08, 0.55, "#8e44ad", "UPS 1500VA", 0.95))
        _cr.add_trace(_box3d(0.35, 0.32, 0.60, 1.15, 1.08, 0.76, "#2980b9", "Industrial Switch (Cisco IE-2000)", 0.95))
        _cr.add_trace(_box3d(0.35, 0.32, 0.78, 1.15, 1.08, 0.94, "#e74c3c", "Industrial Firewall", 0.95))
        _cr.add_trace(_box3d(0.35, 0.32, 0.96, 1.15, 1.08, 1.12, "#f39c12", "OPC-UA Gateway (KEPServerEX)", 0.95))
        _cr.add_trace(_box3d(0.35, 0.32, 1.20, 1.15, 1.08, 1.55, "#16a085", "INTEGRATE Edge Server (ADVANTECH)", 0.98))
        _cr.add_trace(_box3d(0.32, 0.30, 1.56, 1.18, 0.34, 1.58, "#00ff88", "Status LED (Online)", 0.9))
        _cr.add_trace(_box3d(2.5, 0.2, 0, 5.2, 1.3, 0.76, "#34495e", "Meja Operator 1", 0.65))
        _cr.add_trace(_box3d(2.6, 0.55, 0, 3.05, 1.15, 0.42, "#2c3e50", "PC Tower Operator 1", 0.85))
        _cr.add_trace(_box3d(3.2, 0.28, 0.76, 4.9, 0.44, 1.58, "#00b4d8", "Monitor 27\" Op. 1", 0.88))
        _cr.add_trace(_box3d(3.8, 1.0, 0.76, 4.3, 1.3, 0.82, "#555", "Keyboard Op. 1", 0.7))
        _cr.add_trace(_box3d(6.0, 0.2, 0, 8.7, 1.3, 0.76, "#34495e", "Meja Operator 2", 0.65))
        _cr.add_trace(_box3d(6.1, 0.55, 0, 6.55, 1.15, 0.42, "#2c3e50", "PC Tower Operator 2", 0.85))
        _cr.add_trace(_box3d(6.8, 0.28, 0.76, 8.5, 0.44, 1.58, "#00b4d8", "Monitor 27\" Op. 2", 0.88))
        _cr.add_trace(_box3d(7.3, 1.0, 0.76, 7.8, 1.3, 0.82, "#555", "Keyboard Op. 2", 0.7))
        _cr.add_trace(_box3d(1.2, 0.55, 1.98, 8.7, 0.75, 2.04, "#7f8c8d", "Cable Tray LAN (Plafon)", 0.60))
        _cr.add_trace(_line3d([0.75, 0.75, 4.05, 4.05], [0.65, 0.65, 0.65, 0.36],
                              [1.48, 2.01, 2.01, 0.76], "#f39c12", "Kabel LAN ke WS 1"))
        _cr.add_trace(_line3d([0.75, 0.75, 7.25, 7.25], [0.65, 0.65, 0.65, 0.36],
                              [1.48, 2.01, 2.01, 0.76], "#e67e22", "Kabel LAN ke WS 2"))
        _cr.add_trace(_line3d([0.30, -0.5, -0.5], [0.70, 0.70, 0.70],
                              [1.10, 1.10, 1.10], "#8e44ad", "OPC-UA ke Exaquantum/DCS"))
        _cr.add_trace(_line3d([0.75, 0.75], [1.10, 7.0], [0.06, 0.06],
                              "#e74c3c", "Kabel Power (PLN + UPS)"))
        _cr.add_trace(_label3d([0.75, 4.05, 7.25, -0.8], [2.25, 0.3, 0.3, 0.70],
                               [2.30, 1.70, 1.70, 1.30],
                               ["SERVER RACK", "WORKSTATION 1", "WORKSTATION 2", "ke DCS"],
                               color="#00b4d8", size=12))
        _cr.update_layout(
            scene=dict(**_dark_scene, camera=dict(eye=dict(x=1.6, y=-2.2, z=1.3)),
                       aspectratio=dict(x=2.2, y=1.5, z=0.7)),
            paper_bgcolor="#0e1117", height=520, margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(bgcolor="#1a1f2e", bordercolor="#444",
                        font=dict(color="white", size=10), x=0.0, y=1.0),
            title=dict(text="3D Mock-up — Ruang Kontrol INTEGRATE (Control Room)",
                       font=dict(color="#00b4d8", size=13))
        )
        st.plotly_chart(_cr, use_container_width=True)
        st.caption("Klik & drag untuk rotasi | Scroll zoom | Klik nama di legend untuk sembunyikan item")

    if is_pump:
        with st.expander("🔩  Ilustrasi 3D — Pemasangan Sensor pada P-1001A (DMF Pump)", expanded=True):
            _eq = go.Figure()
            _eq.add_trace(_box3d(-0.2, 0.0, 0.0,  4.3, 2.5, 0.45, "#7f8c8d", "P-1001A: Baseplate", 0.55))
            _eq.add_trace(_box3d(0.0,  0.3, 0.45, 2.3, 2.2, 1.42, "#2c3e50", "P-1001A: Casing Pompa", 0.78))
            _eq.add_trace(_box3d(2.3,  0.4, 0.45, 4.1, 2.1, 1.32, "#1a252f", "P-1001A: Motor Listrik", 0.82))
            _eq.add_trace(_box3d(-1.2, 0.8, 0.60, 0.0,  1.3, 1.10, "#3d566e", "P-1001A: Suction Pipe", 0.80))
            _eq.add_trace(_box3d(0.8,  2.2, 0.88, 1.4,  2.80, 1.42, "#3d566e", "P-1001A: Discharge Pipe", 0.80))
            _eq.add_trace(_box3d(2.1,  1.6, 0.85, 2.35, 2.0, 1.25, "#555", "P-1001A: Coupling Guard", 0.60))
            _eq.add_trace(_box3d(-1.80, 0.82, 0.58, -1.20, 1.28, 1.08, "#f39c12", "FT-201: Coriolis Flow Meter", 1.0))
            _eq.add_trace(_box3d(-0.85, 0.90, 0.95, -0.58, 1.22, 1.18, "#e74c3c", "PT-201: Suction Press Tx", 1.0))
            _eq.add_trace(_box3d(0.85,  2.22, 1.12,  1.12, 2.55, 1.38, "#e74c3c", "PT-202: Discharge Press Tx", 1.0))
            _eq.add_trace(_box3d(0.1,   0.10, 1.28,  0.42, 0.38, 1.50, "#3498db", "TT-201: Temperature RTD", 1.0))
            _eq.add_trace(_box3d(3.10,  0.25, 1.10,  3.40, 0.55, 1.35, "#9b59b6", "IT-201: Motor Current Sensor", 1.0))
            _eq.add_trace(_box3d(0.10,  2.05, 0.88,  0.38, 2.32, 1.10, "#27ae60", "PT-203: Seal Pressure Tx", 1.0))
            _eq.add_trace(_box3d(3.60, 2.20, 0.80,  4.20, 2.70, 1.35, "#f1c40f", "JB-P1001A: Junction Box", 0.90))
            for _xs, _ys, _zs in [
                ([-1.50, 3.90], [1.05, 2.45], [0.83, 1.08]),
                ([-0.72, 3.90], [1.06, 2.45], [1.07, 1.08]),
                ([0.99,  3.90], [2.39, 2.45], [1.25, 1.08]),
                ([0.26,  3.90], [0.24, 2.45], [1.39, 1.08]),
            ]:
                _eq.add_trace(go.Scatter3d(x=_xs, y=_ys, z=_zs, mode="lines",
                                           line=dict(color="#aaaaaa", width=1.5),
                                           showlegend=False, hoverinfo="skip"))
            _eq.add_trace(_label3d(
                [1.15, -1.50, -0.72, 0.99, 0.26, 3.25, 3.25, 3.90],
                [0.25, 1.05, 1.06, 2.39, 0.24, 0.30, 0.30, 2.60],
                [1.55, 1.22, 1.32, 1.52, 1.65, 1.22, 0.85, 1.38],
                ["P-1001A", "FT-201", "PT-201", "PT-202", "TT-201", "IT-201", "PT-203", "JB"],
                color="#00b4d8", size=10))
            _eq.update_layout(
                scene=dict(**_dark_scene, camera=dict(eye=dict(x=1.8, y=-2.8, z=1.6)),
                           aspectratio=dict(x=2.0, y=1.5, z=0.55)),
                paper_bgcolor="#0e1117", height=520, margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(bgcolor="#1a1f2e", bordercolor="#444",
                            font=dict(color="white", size=9), x=0.0, y=1.0, itemsizing="constant"),
                title=dict(text="3D Mock-up — Pemasangan Sensor P-1001A (DMF Pump)",
                           font=dict(color="#00b4d8", size=13))
            )
            st.plotly_chart(_eq, use_container_width=True)
            st.markdown("**Kode Warna Sensor:**")
            _lcols = st.columns(5)
            for _col, (_clr, _lbl) in zip(_lcols, [
                ("#e74c3c", "Pressure Transmitter (PT)"), ("#f39c12", "Flow Meter (FT)"),
                ("#3498db", "Temperature RTD (TT)"), ("#9b59b6", "Motor Current (IT)"),
                ("#27ae60", "Seal Pressure (PT)"),
            ]):
                with _col:
                    st.markdown(
                        f"<div style='background:{_clr}33; border:2px solid {_clr}; "
                        f"border-radius:6px; padding:6px 8px; text-align:center; "
                        f"color:white; font-size:12px'>{_lbl}</div>",
                        unsafe_allow_html=True
                    )
            st.caption("Junction Box kuning = titik pengumpulan kabel sinyal sebelum masuk ke DCS panel.")

    else:
        with st.expander("🔩  Ilustrasi 3D — Pemasangan Sensor pada C-1001B (BCL305 Compressor)", expanded=True):
            _eq = go.Figure()
            _eq.add_trace(_box3d(-0.3, 0.5, 0.0,  7.2, 3.5, 0.45, "#7f8c8d", "C-1001B: Baseplate", 0.55))
            _eq.add_trace(_box3d(0.0,  1.0, 0.45, 4.8, 3.0, 1.85, "#2c3e50", "C-1001B: Casing Kompresor (BCL305)", 0.78))
            _eq.add_trace(_box3d(-1.0, 1.7, 0.85, 0.0, 2.3, 1.45, "#3d566e", "C-1001B: Inlet Nozzle", 0.80))
            _eq.add_trace(_box3d(4.8,  1.7, 1.30, 5.6, 2.3, 1.85, "#3d566e", "C-1001B: Discharge Nozzle", 0.80))
            _eq.add_trace(_box3d(4.8,  1.0, 0.45, 7.0, 3.0, 1.65, "#1a252f", "C-1001B: Motor Penggerak", 0.82))
            _eq.add_trace(_box3d(4.5,  1.6, 0.85, 4.8, 2.4, 1.35, "#555",    "C-1001B: Coupling Guard",  0.60))
            _eq.add_trace(_box3d(-0.8, 1.90, 1.10, -0.50, 2.25, 1.38, "#e74c3c", "PT-101: Suction Press. Tx", 1.0))
            _eq.add_trace(_box3d(4.82, 1.90, 1.42,  5.12, 2.25, 1.70, "#e74c3c", "PT-102: Discharge Press. Tx", 1.0))
            _eq.add_trace(_box3d(-1.80, 1.75, 0.82, -1.00, 2.25, 1.42, "#f39c12", "FT-101: Coriolis Flow Meter", 1.0))
            _eq.add_trace(_box3d(-0.75, 1.55, 0.88, -0.48, 1.80, 1.10, "#3498db", "TT-101: Suction Temp RTD", 1.0))
            _eq.add_trace(_box3d(4.82, 1.55, 1.08,  5.12, 1.80, 1.30, "#3498db", "TT-102: Discharge Temp RTD", 1.0))
            _eq.add_trace(_box3d(4.82, 2.90, 1.00,  5.12, 3.20, 1.28, "#9b59b6", "ST-101: Speed Sensor", 1.0))
            _eq.add_trace(_box3d(2.00, 3.00, 1.55,  2.40, 3.30, 1.85, "#27ae60", "VT-101: Vibration Probe", 1.0))
            _eq.add_trace(_box3d(6.50, 3.00, 1.00,  7.10, 3.50, 1.55, "#f1c40f", "JB-C1001B: Junction Box", 0.90))
            for _xs, _ys, _zs in [
                ([-0.65, 6.80], [2.075, 3.25], [1.24, 1.28]),
                ([4.97, 6.80],  [2.075, 3.25], [1.56, 1.28]),
                ([-1.40, 6.80], [2.00,  3.25], [1.12, 1.28]),
                ([-0.615,6.80], [1.675, 3.25], [0.99, 1.28]),
            ]:
                _eq.add_trace(go.Scatter3d(x=_xs, y=_ys, z=_zs, mode="lines",
                                           line=dict(color="#aaaaaa", width=1.5),
                                           showlegend=False, hoverinfo="skip"))
            _eq.add_trace(_label3d(
                [2.4, -0.65, 4.97, -1.4, -0.615, 2.20, 5.10, 6.80],
                [0.50, 2.075, 2.075, 2.00, 1.675, 3.15, 3.25, 3.35],
                [2.10, 1.45, 1.82, 1.55, 1.22, 2.00, 1.20, 1.65],
                ["C-1001B\n(BCL305)", "PT-101", "PT-102", "FT-101", "TT-101", "VT-101", "ST-101", "JB"],
                color="#00b4d8", size=10))
            _eq.update_layout(
                scene=dict(**_dark_scene, camera=dict(eye=dict(x=1.8, y=-2.8, z=1.6)),
                           aspectratio=dict(x=2.0, y=1.5, z=0.55)),
                paper_bgcolor="#0e1117", height=540, margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(bgcolor="#1a1f2e", bordercolor="#444",
                            font=dict(color="white", size=9), x=0.0, y=1.0, itemsizing="constant"),
                title=dict(text="3D Mock-up — Pemasangan Sensor C-1001B (BCL305 Compressor)",
                           font=dict(color="#00b4d8", size=13))
            )
            st.plotly_chart(_eq, use_container_width=True)
            st.markdown("**Kode Warna Sensor:**")
            _lcols = st.columns(5)
            for _col, (_clr, _lbl) in zip(_lcols, [
                ("#e74c3c", "Pressure Transmitter (PT)"), ("#f39c12", "Flow Meter (FT)"),
                ("#3498db", "Temperature RTD (TT)"), ("#9b59b6", "Speed Sensor (ST)"),
                ("#27ae60", "Vibration Probe (VT)"),
            ]):
                with _col:
                    st.markdown(
                        f"<div style='background:{_clr}33; border:2px solid {_clr}; "
                        f"border-radius:6px; padding:6px 8px; text-align:center; "
                        f"color:white; font-size:12px'>{_lbl}</div>",
                        unsafe_allow_html=True
                    )
            st.caption("Junction Box kuning = titik pengumpulan kabel sinyal sebelum masuk ke DCS panel.")