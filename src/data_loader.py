"""
INTEGRATE - Data Loader Module
Handles data ingestion from CSV/Excel (DCS export from Exaquantum)
and stores to SQLite database.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

DB_PATH = os.getenv("DB_PATH", "integrate.db")


# ─────────────────────────────────────────────
# DATABASE INITIALIZATION
# ─────────────────────────────────────────────


def init_database():
    """Create tables if not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Pump operational data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pump_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            equipment_id TEXT NOT NULL,
            flow_rate REAL,
            suction_pressure REAL,
            discharge_pressure REAL,
            temperature REAL,
            motor_current REAL,
            seal_pressure REAL,
            head REAL,
            power REAL,
            efficiency REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Compressor operational data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compressor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            equipment_id TEXT NOT NULL,
            suction_pressure REAL,
            discharge_pressure REAL,
            suction_temperature REAL,
            discharge_temperature REAL,
            inlet_flow REAL,
            shaft_power REAL,
            pressure_ratio REAL,
            polytropic_head REAL,
            surge_margin REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Anomaly log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            equipment_id TEXT NOT NULL,
            equipment_type TEXT NOT NULL,
            anomaly_type TEXT,
            ml_class TEXT,
            confidence REAL,
            mae_value REAL,
            threshold REAL,
            status TEXT,
            recommended_action TEXT,
            notified INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


# ─────────────────────────────────────────────
# DATA LOADING — PUMP
# ─────────────────────────────────────────────


def load_pump_csv(filepath: str, equipment_id: str = "P-9027A") -> pd.DataFrame:
    """
    Load pump operational data from CSV/Excel.
    Compatible with Exaquantum export format (JTB & DMF).

    Parameters
    ----------
    filepath : str
        Path to CSV or Excel file
    equipment_id : str
        Equipment tag (e.g., P-9027A, P-1001A)

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed pump dataframe
    """
    print(f"[DataLoader] Loading pump data: {filepath}")

    # Support CSV and Excel
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Standardize column names (lowercase, strip spaces)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Try to detect timestamp column
    ts_candidates = ["timestamp", "datetime", "time", "date", "waktu"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col:
        df["timestamp"] = pd.to_datetime(df[ts_col])
        df.drop(columns=[ts_col], errors="ignore", inplace=True)
    else:
        df["timestamp"] = pd.date_range(
            start=datetime.now(), periods=len(df), freq="15min"
        )

    df["equipment_id"] = equipment_id

    # ── Handle Missing Values ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )
    df.dropna(inplace=True)

    # ── IQR Outlier Removal ──
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~(
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    df = df[mask].reset_index(drop=True)

    print(f"[DataLoader] Pump data loaded: {len(df)} records after cleaning")
    return df


def load_compressor_csv(filepath: str, equipment_id: str = "C-1001A") -> pd.DataFrame:
    """
    Load compressor operational data from CSV/Excel.
    Includes thermodynamic parameter calculation (ASME PTC 10).
    """
    print(f"[DataLoader] Loading compressor data: {filepath}")

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Timestamp
    ts_candidates = ["timestamp", "datetime", "time", "date", "waktu"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col:
        df["timestamp"] = pd.to_datetime(df[ts_col])
        df.drop(columns=[ts_col], errors="ignore", inplace=True)
    else:
        df["timestamp"] = pd.date_range(
            start=datetime.now(), periods=len(df), freq="15min"
        )

    df["equipment_id"] = equipment_id

    # ── Calculate thermodynamic parameters if not present ──
    # Pressure Ratio (dimensionless)
    if "pressure_ratio" not in df.columns:
        if "discharge_pressure" in df.columns and "suction_pressure" in df.columns:
            df["pressure_ratio"] = df["discharge_pressure"] / df["suction_pressure"]

    # Polytropic Head (ASME PTC 10) — simplified calculation
    # Hp = (n/(n-1)) * (R/M) * T_suc * [(P_dis/P_suc)^((n-1)/n) - 1]
    # Using n=1.3 (typical for natural gas), R=8314 J/kmol·K, M=18 kg/kmol (approx)
    if "polytropic_head" not in df.columns:
        if all(c in df.columns for c in ["suction_temperature", "pressure_ratio"]):
            n = 1.3
            R = 8314
            M = 18
            T_suc_K = df["suction_temperature"] + 273.15  # °F → K conversion if needed
            df["polytropic_head"] = (
                (n / (n - 1))
                * (R / M)
                * T_suc_K
                * (df["pressure_ratio"] ** ((n - 1) / n) - 1)
                / 1000
            )  # kJ/kg

    # ── Missing Values & Outliers ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )
    df.dropna(inplace=True)

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~(
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    df = df[mask].reset_index(drop=True)

    print(f"[DataLoader] Compressor data loaded: {len(df)} records after cleaning")
    return df


# ─────────────────────────────────────────────
# SAVE TO DATABASE
# ─────────────────────────────────────────────


def save_pump_to_db(df: pd.DataFrame):
    """Save pump dataframe to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df_save = df.copy()
    df_save["timestamp"] = df_save["timestamp"].astype(str)
    df_save.to_sql("pump_data", conn, if_exists="append", index=False)
    conn.close()
    print(f"[DB] Saved {len(df_save)} pump records to database.")


def save_compressor_to_db(df: pd.DataFrame):
    """Save compressor dataframe to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df_save = df.copy()
    df_save["timestamp"] = df_save["timestamp"].astype(str)
    df_save.to_sql("compressor_data", conn, if_exists="append", index=False)
    conn.close()
    print(f"[DB] Saved {len(df_save)} compressor records to database.")


def log_anomaly(
    equipment_id: str,
    equipment_type: str,
    anomaly_type: str,
    ml_class: str,
    confidence: float,
    mae: float,
    threshold: float,
    status: str,
    action: str,
):
    """Insert anomaly detection result to anomaly_log table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO anomaly_log 
        (timestamp, equipment_id, equipment_type, anomaly_type, ml_class,
         confidence, mae_value, threshold, status, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            datetime.now().isoformat(),
            equipment_id,
            equipment_type,
            anomaly_type,
            ml_class,
            confidence,
            mae,
            threshold,
            status,
            action,
        ),
    )
    conn.commit()
    conn.close()


def get_anomaly_log(limit: int = 50) -> pd.DataFrame:
    """Retrieve latest anomaly log from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM anomaly_log ORDER BY created_at DESC LIMIT {limit}", conn
    )
    conn.close()
    return df


def get_latest_pump_data(equipment_id: str, n: int = 100) -> pd.DataFrame:
    """Get latest N records for a specific pump from DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM pump_data WHERE equipment_id=? ORDER BY timestamp DESC LIMIT {n}",
        conn,
        params=(equipment_id,),
    )
    conn.close()
    return df.sort_values("timestamp")


def get_latest_compressor_data(equipment_id: str, n: int = 100) -> pd.DataFrame:
    """Get latest N records for a specific compressor from DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM compressor_data WHERE equipment_id=? ORDER BY timestamp DESC LIMIT {n}",
        conn,
        params=(equipment_id,),
    )
    conn.close()
    return df.sort_values("timestamp")
