# INTEGRATE
**Intelligent Real-Time Rotating Equipment Anomaly Detection & Performance Monitoring System**

Sistem Pemantauan Pompa & Kompresor Terintegrasi Berbasis Deep Learning, Rule-Based Analytics, dan Real-Time Performance Curve

---

## Digital Hackathon AI/ML Hulu Migas 2026

**Team ANTEK ITS:**
- Kania Indah Ramadhan (Lead)
- Nurussyawal Latansa Fitri
- Fauzan Randy Susanto
- Novansyah Bagus Pramudya Hanafi
- Akmal Alfarizky

---

## Persyaratan Sistem

| Komponen | Versi |
|---|---|
| **Python** | **3.11.x** (WAJIB — TensorFlow 2.15 tidak support Python 3.12+) |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| RAM | Minimal 8 GB |

---

## Quick Start

### Windows

```bash
# 1. Clone repository
git clone https://github.com/kaniair/INTEGRATE-ANTEK-ITS.git
cd INTEGRATE-ANTEK-ITS

# 2. Pastikan Python 3.11 terinstall
# Download: https://www.python.org/downloads/release/python-3119/
python --version   # harus menampilkan Python 3.11.x

# 3. Buat virtual environment dengan Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Jalankan dashboard
streamlit run app.py
```

### Linux / macOS (Intel)

```bash
git clone https://github.com/kaniair/INTEGRATE-ANTEK-ITS.git
cd INTEGRATE-ANTEK-ITS

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### macOS (Apple Silicon M1/M2/M3)

TensorFlow di Apple Silicon memerlukan package khusus:

```bash
git clone https://github.com/kaniair/INTEGRATE-ANTEK-ITS.git
cd INTEGRATE-ANTEK-ITS

python3.11 -m venv .venv
source .venv/bin/activate

# Install tensorflow versi Apple Silicon
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal

# Install sisanya (skip tensorflow di requirements.txt)
pip install streamlit==1.35.0 pandas==2.1.4 numpy==1.26.4 scipy==1.12.0 \
    scikit-learn==1.4.0 xgboost==2.0.3 imbalanced-learn==0.12.0 \
    optuna==3.5.0 sqlalchemy==2.0.27 plotly==5.18.0 \
    openpyxl==3.1.2 python-dotenv==1.0.1 rule-engine==4.4.0

streamlit run app.py
```

Buka browser → **http://localhost:8501**

---

## Struktur Project

```
INTEGRATE-ANTEK-ITS/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt
├── src/
│   ├── ml_engine.py                # BiLSTM Autoencoder + XGBoost
│   ├── curve_engine.py             # Rule-based 8-zone performance engine
│   ├── data_loader.py              # Data loading & preprocessing
│   └── alerting.py                 # Email alert & recommendations
├── curves/
│   ├── compressor_C1001B_curves.json   # C-1001B design specs & op stats
│   ├── compressor_C1001B_fitted.json   # Fitted polynomials from DCS data
│   └── pump_curves.json                # Pump performance curves
└── data/
    └── datacompressor_C1001B/
        └── COMPRESSOR DONGGI FIX_2 - Copy.xlsx  # DCS data Exaquantum
```

---

## Fitur Utama

### Equipment yang Dimonitor
- **P-1001A** — Pompa (Centrifugal Pump)
- **C-1001B** — Kompresor Sentrifugal Donggi BCL305

### Tab Dashboard
1. **📊 Dashboard Real-Time** — KPI metrics, time series, anomaly detection
2. **📈 Kurva Performa** — 4 performance curve charts dengan speed lines A–E:
   - Polytropic Head vs Standard Inlet Flow
   - Pressure Ratio vs Standard Inlet Flow
   - Discharge Pressure vs Standard Inlet Flow
   - Shaft Power vs Standard Inlet Flow
   - Filter rentang tanggal + analisis per grafik + kesimpulan akhir
3. **🤖 Deteksi Anomali** — BiLSTM Autoencoder + Rule Engine
4. **📋 Laporan** — Export & history

### Klasifikasi Zona Kompresor C-1001B (API 617 / ASME PTC 10)
| Zona | Keterangan |
|---|---|
| Z1 — Surge Zone | KRITIS: flow di bawah surge line |
| Z2 — Protection Zone | WASPADA: mendekati surge |
| Z3 — Zona Operasi Optimal | Antara SpeedB & Protection Line |
| Z4 — Zona Operasi Normal | Antara SpeedC & SpeedB |
| Z5 — Zona Kecepatan Rendah | Antara SpeedD & SpeedC |
| Z6 — Zona Kecepatan Minimum | Antara SpeedE & SpeedD |
| Z7 — Di Bawah Envelope | Di bawah SpeedE |
| Z8 — Stonewall / Choke | Flow terlalu tinggi |

---

## Konfigurasi Email Alert (Opsional)

Buat file `recruitment.env` di root folder (jangan di-commit):

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_RECIPIENTS=engineer1@company.com,engineer2@company.com
```

Tanpa file ini, dashboard tetap berjalan normal — hanya fitur email alert yang tidak aktif.

---

## Troubleshooting

**`ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0`**
→ Pastikan Python 3.11 yang aktif: `python --version`
→ Windows: gunakan `py -3.11 -m venv .venv`

**`ModuleNotFoundError: No module named 'streamlit'`**
→ Virtual environment belum aktif: `.venv\Scripts\activate` (Windows) atau `source .venv/bin/activate` (Linux/Mac)

**Port 8501 sudah digunakan**
```bash
streamlit run app.py --server.port 8502
```

**Data DCS tidak terbaca**
→ File Excel sudah disertakan di repo. Pastikan path `data/datacompressor_C1001B/COMPRESSOR DONGGI FIX_2 - Copy.xlsx` ada setelah clone.
