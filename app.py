"""
INTEGRATE — Intelligent Real-Time Rotating Equipment
Anomaly Detection & Performance Monitoring System

Main entry point: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import internal modules
from src.data_loader import (
    init_database, load_pump_csv, load_compressor_csv,
    save_pump_to_db, save_compressor_to_db,
    get_anomaly_log, get_latest_pump_data, get_latest_compressor_data,
    log_anomaly
)
from src.ml_engine import detect_anomaly, classify_anomaly, PUMP_FEATURES, COMP_FEATURES
from src.curve_engine import (
    load_pump_curves, load_compressor_curves,
    check_pump_performance, check_compressor_performance,
    PumpZone, CompressorZone
)
from src.alerting import send_anomaly_alert, get_recommended_action

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="INTEGRATE Monitoring System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# Load curves
pump_curves = load_pump_curves()
comp_curves = load_compressor_curves()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Pertamina.svg/200px-Pertamina.svg.png",
             width=120)
    st.title("⚙️ INTEGRATE")
    st.caption("Intelligent Rotating Equipment Monitor")
    st.divider()

    # Equipment selection
    st.subheader("📋 Pilih Equipment")
    equipment_type = st.selectbox(
        "Tipe Equipment",
        ["Pompa Sentrifugal", "Kompresor Sentrifugal"]
    )

    if equipment_type == "Pompa Sentrifugal":
        equipment_id = st.selectbox(
            "Equipment ID",
            ["P-1001A (DMF)", "P-1001B (DMF)"]
        )

    equipment_tag = equipment_id.split(" ")[0]  # e.g., "P-9027A"
    st.divider()

    # File upload
    st.subheader("📂 Upload Data CSV")
    uploaded_file = st.file_uploader(
        "Upload file CSV/Excel (ekspor Exaquantum)",
        type=["csv", "xlsx", "xls"],
        help="Format: timestamp, parameter operasi"
    )

    # Date range
    st.subheader("📅 Rentang Waktu")
    date_from = st.date_input("Dari", value=datetime.now() - timedelta(days=7))
    date_to = st.date_input("Sampai", value=datetime.now())

    run_analysis = st.button("🔍 Jalankan Analisis", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"🕐 Terakhir refresh: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.title("⚙️ INTEGRATE — Monitoring Rotating Equipment")
st.caption(f"PT Pertamina EP Cepu | Digital Hackathon AI/ML Hulu Migas 2026")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Utama",
    "📈 Kurva Performa",
    "🔍 Anomaly Detection",
    "📋 Log Anomali"
])


# ═══════════════════════════════════════════
# TAB 1 — DASHBOARD UTAMA
# ═══════════════════════════════════════════

with tab1:
    # Status Cards
    st.subheader("Status Equipment Real-Time")

    col1, col2, col3 = st.columns(3)

    # Sample data (akan diganti data real dari DB)
    with col1:
        st.metric(
            label="P-9027A (JTB)",
            value="NORMAL ✅",
            delta="MAE: 0.072 < 0.108",
            delta_color="normal"
        )
        st.caption("Zone: POR | Flow: 368 kBPD")

    with col2:
        st.metric(
            label="P-9027B (JTB)",
            value="WARNING ⚠️",
            delta="MAE: 0.119 > 0.108",
            delta_color="inverse"
        )
        st.caption("Zone: AOR | Cek mechanical seal")

    with col3:
        st.metric(
            label="C-1001B (DMF)",
            value="NORMAL ✅",
            delta="Surge Margin: 34%",
            delta_color="normal"
        )
        st.caption("Flow: 57.3 MMSCFD | rp: 1.82")

    st.divider()

    # Time-series chart
    st.subheader(f"📈 Parameter vs Waktu — {equipment_tag}")

    if uploaded_file is not None and run_analysis:
        # Load uploaded data
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if equipment_type == "Pompa Sentrifugal":
                df = load_pump_csv(tmp_path, equipment_tag)
                save_pump_to_db(df)
                df = detect_anomaly(df, PUMP_FEATURES, f"models/pump/{equipment_tag}")

                # Time series plot
                fig = go.Figure()
                for col in ["flow_rate", "discharge_pressure", "suction_pressure"]:
                    if col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["timestamp"], y=df[col],
                            name=col, mode="lines"
                        ))

                # Mark anomalies
                if "is_anomaly" in df.columns:
                    anomaly_df = df[df["is_anomaly"] == True]
                    if not anomaly_df.empty and "flow_rate" in df.columns:
                        fig.add_trace(go.Scatter(
                            x=anomaly_df["timestamp"],
                            y=anomaly_df["flow_rate"],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="x"),
                            name="⚠️ Anomali"
                        ))

                fig.update_layout(
                    height=400,
                    xaxis_title="Waktu",
                    yaxis_title="Nilai Parameter",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary
                n_anomaly = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Data Points", len(df))
                c2.metric("Anomali Terdeteksi", n_anomaly, delta=f"{n_anomaly/len(df)*100:.1f}%")
                c3.metric("MAE Rata-rata", f"{df['mae'].mean():.4f}" if 'mae' in df.columns else "N/A")

            else:
                df = load_compressor_csv(tmp_path, equipment_tag)
                save_compressor_to_db(df)
                df = detect_anomaly(df, COMP_FEATURES, f"models/comp/{equipment_tag}")

                fig = go.Figure()
                for col in ["inlet_flow", "pressure_ratio", "shaft_power"]:
                    if col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df["timestamp"], y=df[col],
                            name=col, mode="lines"
                        ))
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading data: {e}")
        finally:
            os.unlink(tmp_path)
    else:
        st.info("⬅️ Upload file CSV dan klik 'Jalankan Analisis' untuk melihat data real-time")
        # Demo chart
        x = pd.date_range(start="2025-01-01", periods=100, freq="15min")
        y = 370 + np.random.normal(0, 5, 100)
        fig = go.Figure(go.Scatter(x=x, y=y, name="Flow Rate (kBPD)", line=dict(color="#3498db")))
        fig.update_layout(height=300, title="Demo — Data Simulasi", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — KURVA PERFORMA
# ═══════════════════════════════════════════

with tab2:
    st.subheader("📈 Real-Time Performance Curve Monitoring")

    if equipment_type == "Pompa Sentrifugal":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Head vs Flow Curve**")

            # 1. Definisi range sumbu X (Flow dalam GPM) seperti di Excel
            flow_range = np.linspace(0, 800, 200)

            # 2. Masukkan koefisien polinomial sesuai tabel referensi Excel
            coeff_max = [4e-12, -6e-9, 3e-7, 0.0003, -0.1651, 2353.8]
            coeff_rated = [5e-13, 1e-9, -4e-6, 0.0013, -0.2108, 2146.9]
            coeff_min = [-8e-13, 4e-9, -6e-6, 0.0016, -0.1897, 1858.3]

            # Hitung nilai Y (Head) untuk 3 kurva utama
            head_max = np.polyval(coeff_max, flow_range)
            head_rated = np.polyval(coeff_rated, flow_range)
            head_min = np.polyval(coeff_min, flow_range)

            fig_hf = go.Figure()

            # 3. Plot 3 Kurva Utama dengan warna identik dengan Excel
            fig_hf.add_trace(go.Scatter(x=flow_range, y=head_max, name="Max Diameter", line=dict(color="#4472c4", width=2)))
            fig_hf.add_trace(go.Scatter(x=flow_range, y=head_rated, name="Rated Diameter", line=dict(color="#a5a5a5", width=2)))
            fig_hf.add_trace(go.Scatter(x=flow_range, y=head_min, name="Min Diameter", line=dict(color="#ed7d31", width=2)))

            # 4. Plot Garis Batas (Min Flow, Lower POR, Upper POR)
            # Kita buat dummy nilai Y, lalu hitung persamaan linear X-nya
            y_bounds = np.linspace(1000, 2500, 100)
            x_min_flow = (y_bounds + 1024.5) / 24.97
            x_lower_por = (y_bounds + 759174) / 1811.1
            x_upper_por = (y_bounds + 14539) / 22.626

            # Fungsi cerdas untuk "memotong" garis batas agar tidak bablas 
            # (hanya digambar di antara garis Max Diameter dan Min Diameter)
            def filter_bounds(x_arr, y_arr):
                vx, vy = [], []
                for x, y in zip(x_arr, y_arr):
                    if 0 <= x <= 800:
                        y_maks_batas = np.polyval(coeff_max, x)
                        y_min_batas = np.polyval(coeff_min, x)
                        if y_min_batas <= y <= y_maks_batas:
                            vx.append(x)
                            vy.append(y)
                return vx, vy

            x_mf, y_mf = filter_bounds(x_min_flow, y_bounds)
            x_lp, y_lp = filter_bounds(x_lower_por, y_bounds)
            x_up, y_up = filter_bounds(x_upper_por, y_bounds)

            # Tambahkan garis batas ke plot
            fig_hf.add_trace(go.Scatter(x=x_mf, y=y_mf, name="Min Flow", mode="lines", line=dict(color="#5b9bd5", width=3, dash="dash")))
            fig_hf.add_trace(go.Scatter(x=x_lp, y=y_lp, name="Lower POR", mode="lines", line=dict(color="#70ad47", width=3, dash="dash")))
            fig_hf.add_trace(go.Scatter(x=x_up, y=y_up, name="Upper POR", mode="lines", line=dict(color="#1f3864", width=3, dash="dash")))

            # 5. Plot Titik Operasi Aktual DCS (Dapat disesuaikan dgn variabel DB-mu nanti)
            # Sementara ini sample dari chart Excel
            actual_flows = [368, 370, 422, 425]
            actual_heads = [1965, 2035, 2065, 1960]
            
            fig_hf.add_trace(go.Scatter(
                x=actual_flows, y=actual_heads, mode="markers",
                marker=dict(color="red", size=8),
                name="Actual Point - DCS"
            ))

            # 6. Pengaturan Tata Letak Plotly (Axis & Legend)
            fig_hf.update_layout(
                height=500,
                xaxis_title="Flow (Gpm)",
                yaxis_title="Head - ft",
                xaxis=dict(range=[0, 850]),
                yaxis=dict(range=[1000, 2500]),
                hovermode="closest",
                legend=dict(
                    yanchor="top", y=0.99, 
                    xanchor="left", x=1.02 # Posisi legend di sebelah kanan grafik
                ),
                margin=dict(r=150) # Beri ruang di kanan untuk legend
            )
            
            st.plotly_chart(fig_hf, use_container_width=True)

        with col2:
            st.markdown("**NPSH Available vs NPSH Required**")

            # 1. Definisi range sumbu X (Flow dalam GPM)
            flow_range_npsh = np.linspace(0, 700, 100)

            # 2. Masukkan koefisien polinomial NPSHR sesuai tabel
            coeff_npshr = [-2e-13, 3e-10, -2e-7, 0.0001, -0.0032, 7.8808]
            npshr_curve = np.polyval(coeff_npshr, flow_range_npsh)

            fig_npsh = go.Figure()

            # 3. Plot Kurva NPSHR (Garis Biru seperti Excel)
            fig_npsh.add_trace(go.Scatter(
                x=flow_range_npsh, y=npshr_curve,
                name="NPSHR", 
                line=dict(color="#4472c4", width=3)
            ))

            # 4. Plot Titik Aktual NPSHa (Titik Oranye)
            # Ini adalah data sampel yang meniru pola sebaran di Excel.
            # Nantinya, list ini diganti dengan data aktual yang ditarik dari DCS/DB.
            actual_flows_npsh = [360, 362, 365, 368, 370, 420, 422, 425, 510, 515, 518]
            actual_npsha = [362, 365, 375, 370, 372, 345, 350, 352, 325, 335, 340]

            fig_npsh.add_trace(go.Scatter(
                x=actual_flows_npsh, y=actual_npsha, 
                mode="markers",
                name="NPSHa",
                marker=dict(
                    color="#fbbc05",          # Warna isian oranye kekuningan
                    size=8, 
                    line=dict(color="#e67c22", width=2) # Border titik warna oranye tua
                )
            ))

            # 5. Pengaturan Tata Letak Plotly
            fig_npsh.update_layout(
                height=500,
                xaxis_title="Flow (Gpm)",
                yaxis_title="NPSH (ft)",
                xaxis=dict(range=[0, 750]),
                yaxis=dict(range=[0, 450]),
                hovermode="closest",
                legend=dict(
                    yanchor="top", y=0.99, 
                    xanchor="left", x=1.02 # Posisi legend di sebelah kanan
                ),
                margin=dict(r=120) # Ruang ekstra di kanan untuk legend
            )

            st.plotly_chart(fig_npsh, use_container_width=True)

    else:
        # Compressor curves
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Standard Inlet Flow vs Pressure Ratio**")
            speed_lines = comp_curves["flow_pressure_ratio"]["speed_lines"]
            surge_flow = comp_curves["flow_pressure_ratio"]["surge_flow"]

            fig_comp = go.Figure()
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]
            for i, (speed, data) in enumerate(speed_lines.items()):
                fig_comp.add_trace(go.Scatter(
                    x=data["flow"], y=data["pr"],
                    name=f"Speed {speed}", mode="lines+markers",
                    line=dict(color=colors[i % 3])
                ))

            # Surge line (vertical at surge_flow)
            fig_comp.add_vline(x=surge_flow, line_color="red", line_width=2,
                annotation_text="⛔ Surge Line")
            fig_comp.add_vline(x=surge_flow * 1.1, line_color="orange",
                line_dash="dash", annotation_text="⚠️ Protection Line")

            # Current operating point
            fig_comp.add_trace(go.Scatter(
                x=[57.3], y=[1.82], mode="markers",
                marker=dict(color="blue", size=14, symbol="star"),
                name="Titik Operasi Aktual"
            ))
            fig_comp.update_layout(
                height=350,
                xaxis_title="Standard Inlet Flow (MMSCFD)",
                yaxis_title="Pressure Ratio",
                hovermode="x"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col2:
            st.markdown("**Surge Margin Index**")
            surge_margin = 34.0  # sample

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=surge_margin,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Surge Margin Index (%)"},
                delta={"reference": 20},
                gauge={
                    "axis": {"range": [0, 80]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 10], "color": "#e74c3c"},
                        {"range": [10, 20], "color": "#f39c12"},
                        {"range": [20, 80], "color": "#2ecc71"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 10
                    }
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ═══════════════════════════════════════════

with tab3:
    st.subheader("🔍 Anomaly Detection — BiLSTM Autoencoder")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**MAE (Reconstruction Error) Time Series**")
        # Demo MAE chart
        x = pd.date_range("2025-01-01", periods=200, freq="15min")
        mae = np.abs(np.random.normal(0.06, 0.02, 200))
        # Inject some anomalies
        mae[50:55] = 0.15
        mae[130:135] = 0.18

        threshold_val = 0.108

        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=x, y=mae, name="MAE", line=dict(color="#3498db"), fill="tozeroy"
        ))
        fig_mae.add_hline(
            y=threshold_val, line_color="red", line_dash="dash",
            annotation_text=f"Threshold = {threshold_val}"
        )
        # Color anomalies
        anomaly_mask = mae > threshold_val
        fig_mae.add_trace(go.Scatter(
            x=x[anomaly_mask], y=mae[anomaly_mask],
            mode="markers", name="⚠️ Anomali",
            marker=dict(color="red", size=8, symbol="x")
        ))
        fig_mae.update_layout(height=350, hovermode="x unified",
                              xaxis_title="Waktu", yaxis_title="MAE")
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        st.markdown("**Model Info**")
        st.info("""
        **Arsitektur:** BiLSTM Autoencoder
        
        **Encoder:** BiLSTM(32) → BiLSTM(8)
        
        **Decoder:** BiLSTM(8) → BiLSTM(32) → Dense
        
        **Threshold:** μ + 2.33σ
        
        **Target F1:** > 0.85
        
        **Klasifikasi:** XGBoost + SMOTE
        """)

        st.markdown("**Klasifikasi Anomali**")
        labels = ["Normal", "Startup", "Proses", "Nominasi", "Equipment"]
        values = [75, 5, 8, 7, 5]
        fig_pie = px.pie(values=values, names=labels,
                          color_discrete_sequence=["#2ecc71","#3498db","#f39c12","#9b59b6","#e74c3c"])
        fig_pie.update_layout(height=250, showlegend=True, margin=dict(t=0,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — LOG ANOMALI
# ═══════════════════════════════════════════

with tab4:
    st.subheader("📋 Anomaly Log — Riwayat Deteksi")

    # Try to load from DB, otherwise show sample data
    try:
        df_log = get_anomaly_log(limit=50)
        if df_log.empty:
            raise ValueError("Empty DB")
    except:
        # Sample data for demo
        df_log = pd.DataFrame({
            "timestamp": ["14:17 16/04", "13:45 16/04", "11:30 16/04", "08:15 16/04"],
            "equipment_id": ["P-9027B", "P-9025A", "C-1001B", "P-1001A"],
            "anomaly_type": ["Equipment anomali", "Perubahan proses", "Surge warning", "Startup"],
            "ml_class": ["Equipment", "Proses", "Part_Load", "Startup"],
            "confidence": [0.89, 0.82, 0.76, 0.83],
            "status": ["KRITIS", "WARNING", "SELESAI", "INFO"]
        })

    # Color coding
    def color_status(val):
        colors = {
            "KRITIS": "background-color: #e74c3c; color: white",
            "WARNING": "background-color: #f39c12; color: white",
            "SELESAI": "background-color: #2ecc71; color: white",
            "INFO": "background-color: #3498db; color: white"
        }
        return colors.get(val, "")

    styled = df_log.style.applymap(color_status, subset=["status"])
    st.dataframe(styled, use_container_width=True, height=400)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export CSV", use_container_width=True):
            csv = df_log.to_csv(index=False)
            st.download_button("Download CSV", csv, "anomaly_log.csv", "text/csv")
    with col2:
        if st.button("🔄 Refresh Log", use_container_width=True):
            st.rerun()