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
    with st.expander(f"🔎 Data Mentah ({len(df):,} baris | {_date_label})", expanded=False):
        st.info(f"Menampilkan **{len(df):,}** baris. Merah = anomali ({int(df['is_anomaly'].sum())} titik).",
                icon="📅")
        _disp_cols = ["timestamp"] + [c for c in avail_features if c in df.columns]
        if "mae" in df.columns:        _disp_cols += ["mae"]
        if "is_anomaly" in df.columns: _disp_cols += ["is_anomaly"]
        _df_disp = df[_disp_cols].sort_values("timestamp", ascending=False).copy()

        def _color_anomaly(row):
            if row.get("is_anomaly", False):
                return ["background-color: #4a0000; color: #ff6b6b"] * len(row)
            return [""] * len(row)

        st.dataframe(_df_disp.style.apply(_color_anomaly, axis=1),
                     use_container_width=True, hide_index=True)


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
    # [PASTE SELURUH KODE TAB 5 DARI VERSI SEBELUMNYA — TIDAK ADA YANG BERUBAH]
    # Salin dari "with tab5:" sampai akhir file dari kode yang kamu punya sekarang
    pass