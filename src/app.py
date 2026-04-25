"""
INTEGRATE Monitoring System
Streamlit dashboard for pump & compressor anomaly detection.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.curve_engine import (
    check_pump_performance,
    load_pump_curves,
)
from src.data_loader import init_database, load_pump_csv

load_dotenv("recruitment.env")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title=os.getenv("APP_TITLE", "INTEGRATE Monitoring System"),
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────

st.markdown(
    """
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #4c9be8;
    }
    .alert-critical { border-left-color: #e74c3c !important; }
    .alert-warning  { border-left-color: #f39c12 !important; }
    .alert-ok       { border-left-color: #2ecc71 !important; }
    .stMetric label { font-size: 13px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Pertamina.svg/200px-Pertamina.svg.png",
        width=120,
    )
    st.title("INTEGRATE")
    st.caption("Anomaly Detection System")
    st.divider()

    page = st.radio(
        "Navigasi",
        ["Dashboard", "Analisis Kurva", "Deteksi Anomali", "Log & Riwayat"],
        index=0,
    )

    st.divider()
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV Pompa", type=["csv", "xlsx"], key="uploader"
    )
    equipment_id = st.text_input("Equipment ID", value="P-9027A")

    st.divider()
    st.caption("PT Pertamina EP Cepu")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Memuat data...")
def get_data(file_source, eq_id):
    if file_source is not None:
        # Save uploaded file to temp
        import tempfile
        suffix = ".csv" if file_source.name.endswith(".csv") else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_source.getvalue())
            tmp_path = tmp.name
        return load_pump_csv(tmp_path, eq_id)
    else:
        # Default: load sample data
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "datapump_JTB", "sample_pump_JTB.csv",
        )
        if os.path.exists(sample_path):
            return load_pump_csv(sample_path, eq_id)
        return None


df = get_data(uploaded_file, equipment_id)

if df is None:
    st.error("Data tidak ditemukan. Upload file CSV atau pastikan folder data/ ada.")
    st.stop()

PUMP_FEATURES = [
    "flow_rate", "suction_pressure", "discharge_pressure",
    "temperature", "motor_current", "seal_pressure",
]
available_features = [c for c in PUMP_FEATURES if c in df.columns]

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────

if page == "Dashboard":
    st.title("⚙️ Dashboard Monitoring Pompa")
    st.caption(f"Equipment: **{equipment_id}** — {len(df)} data points")

    # ── KPI row ──
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    cols = st.columns(len(available_features))
    labels = {
        "flow_rate": ("Flow Rate", "kBPD"),
        "suction_pressure": ("Suction Pressure", "psi"),
        "discharge_pressure": ("Discharge Pressure", "psi"),
        "temperature": ("Temperature", "°C"),
        "motor_current": ("Motor Current", "A"),
        "seal_pressure": ("Seal Pressure", "psi"),
    }
    for col, feat in zip(cols, available_features):
        label, unit = labels.get(feat, (feat, ""))
        delta = float(latest[feat]) - float(prev[feat])
        col.metric(
            f"{label} ({unit})",
            f"{latest[feat]:.1f}",
            f"{delta:+.2f}",
        )

    st.divider()

    # ── Trend chart ──
    st.subheader("Tren Operasi")
    selected = st.multiselect(
        "Pilih parameter",
        available_features,
        default=available_features[:3],
    )

    if selected:
        fig = go.Figure()
        for feat in selected:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[feat],
                    name=labels.get(feat, (feat, ""))[0],
                    mode="lines",
                )
            )
        fig.update_layout(
            height=400,
            template="plotly_dark",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Data table ──
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df.tail(20), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: ANALISIS KURVA
# ─────────────────────────────────────────────

elif page == "Analisis Kurva":
    st.title("📈 Analisis Kurva Performa Pompa")
    st.caption("Berdasarkan standar API 610 / ASME PTC 8.2")

    curves = load_pump_curves()
    latest = df.iloc[-1]

    # Input manual override
    with st.expander("Override Parameter Manual", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        flow = c1.number_input("Flow Rate (kBPD)", value=float(latest.get("flow_rate", 370)), step=1.0)
        head = c2.number_input("Head (ft)", value=1954.0, step=10.0)
        power = c3.number_input("Power (kW)", value=float(latest.get("motor_current", 145)) * 3.0, step=10.0)
        npsha = c4.number_input("NPSHa (ft)", value=25.0, step=1.0)

    # Use latest data if not overridden
    if "flow" not in st.session_state:
        flow = float(latest.get("flow_rate", 370))
        head_coeff = curves["head_flow"]["coefficients"]
        import numpy as np
        head = float(np.polyval(head_coeff, flow))
        power = float(latest.get("motor_current", 145)) * 3.0
        npsha = 25.0

    status = check_pump_performance(equipment_id, flow, head, power, npsha, curves)

    # Zone badge
    zone_color = {
        "POR": "🟢", "AOR": "🟡",
        "Cavitation": "🔴", "Min Flow": "🔴", "Overload": "🔴",
    }
    badge = zone_color.get(status.zone.value, "⚪")
    st.subheader(f"{badge} Zona Operasi: **{status.zone.value}**")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Flow Rate", f"{status.flow:.1f} kBPD")
    c2.metric("Head", f"{status.head:.1f} ft")
    c3.metric("Efisiensi", f"{status.efficiency:.1f} %")

    # Alerts
    st.subheader("Status & Alert")
    for msg in status.alert_messages:
        if "KRITIS" in msg or "BAHAYA" in msg:
            st.error(msg)
        elif "PERINGATAN" in msg:
            st.warning(msg)
        else:
            st.success(msg)

    # Operating curve chart
    st.subheader("Kurva Head-Flow")
    import numpy as np

    hf = curves["head_flow"]
    rated_flow = hf["rated_flow"]
    coeff = hf["coefficients"]
    flow_range = np.linspace(0.3 * rated_flow, 1.3 * rated_flow, 100)
    head_curve = np.polyval(coeff, flow_range)

    por_min = hf["por_min_flow"] / 100 * rated_flow
    por_max = hf["por_max_flow"] / 100 * rated_flow

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=flow_range, y=head_curve,
        name="Kurva H-Q", line=dict(color="#4c9be8", width=2),
    ))
    fig.add_vrect(x0=por_min, x1=por_max, fillcolor="green", opacity=0.1,
                  annotation_text="POR", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[status.flow], y=[status.head],
        name="Titik Operasi", mode="markers",
        marker=dict(size=14, color="red", symbol="star"),
    ))
    fig.update_layout(
        height=400, template="plotly_dark",
        xaxis_title="Flow Rate (kBPD)", yaxis_title="Head (ft)",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DETEKSI ANOMALI
# ─────────────────────────────────────────────

elif page == "Deteksi Anomali":
    st.title("🤖 Deteksi Anomali — BiLSTM Autoencoder")

    try:
        from src.ml_engine import detect_anomaly, PUMP_FEATURES as ML_FEATURES
        model_dir = "models/pump"

        if not os.path.exists(os.path.join(model_dir, "threshold.json")):
            st.info(
                "Model belum dilatih. Jalankan training terlebih dahulu.\n\n"
                "**Cara training:**\n```python\n"
                "from src.ml_engine import train_autoencoder, PUMP_FEATURES\n"
                "from src.data_loader import load_pump_csv\n"
                "df = load_pump_csv('data/datapump_JTB/sample_pump_JTB.csv')\n"
                "train_autoencoder(df, PUMP_FEATURES, 'models/pump', epochs=50)\n"
                "```"
            )
        else:
            with st.spinner("Menjalankan deteksi anomali..."):
                feat_cols = [c for c in ML_FEATURES if c in df.columns]
                df_result = detect_anomaly(df.copy(), feat_cols, model_dir)

            anomaly_count = df_result["is_anomaly"].sum()
            total = len(df_result)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Data Points", total)
            c2.metric("Anomali Terdeteksi", anomaly_count,
                      delta=f"{anomaly_count/total*100:.1f}%")
            c3.metric("Status", "⚠️ ADA ANOMALI" if anomaly_count > 0 else "✅ NORMAL")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_result["timestamp"], y=df_result["mae"],
                name="MAE Score", line=dict(color="#4c9be8"),
            ))
            fig.add_hline(
                y=df_result["threshold"].iloc[0],
                line_dash="dash", line_color="red",
                annotation_text="Threshold",
            )
            fig.update_layout(
                height=350, template="plotly_dark",
                xaxis_title="Waktu", yaxis_title="MAE Score",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            if anomaly_count > 0:
                st.subheader("Data Point Anomali")
                st.dataframe(
                    df_result[df_result["is_anomaly"]].tail(10),
                    use_container_width=True,
                )

    except ImportError as e:
        st.warning(
            f"TensorFlow tidak tersedia di environment ini: `{e}`\n\n"
            "Install TensorFlow (Python 3.8–3.11) untuk menggunakan fitur ini."
        )

# ─────────────────────────────────────────────
# PAGE: LOG & RIWAYAT
# ─────────────────────────────────────────────

elif page == "Log & Riwayat":
    st.title("📋 Log Anomali & Riwayat")

    try:
        init_database()
        from src.data_loader import get_anomaly_log
        log_df = get_anomaly_log(limit=100)

        if log_df.empty:
            st.info("Belum ada log anomali. Log akan muncul setelah deteksi anomali berjalan.")
        else:
            st.metric("Total Entri Log", len(log_df))
            fig = px.bar(
                log_df.groupby("anomaly_type").size().reset_index(name="count"),
                x="anomaly_type", y="count",
                title="Distribusi Tipe Anomali",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(log_df, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal memuat log: {e}")
