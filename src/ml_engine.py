"""
INTEGRATE - ML Engine Module
BiLSTM Autoencoder for anomaly detection + XGBoost classifier.
Architecture based on Wei et al. (2022) with enhancements.
"""

import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PUMP_FEATURES = [
    "flow_rate",
    "suction_pressure",
    "discharge_pressure",
    "temperature",
    "motor_current",
    "seal_pressure",
]

COMP_FEATURES = [
    "suction_pressure",
    "discharge_pressure",
    "suction_temperature",
    "discharge_temperature",
    "inlet_flow",
    "shaft_power",
    "pressure_ratio",
]

SEQUENCE_LENGTH = 10  # timesteps per window

ANOMALY_CLASSES = {
    "pump": ["Normal", "Startup", "Proses", "Nominasi", "Equipment"],
    "comp": ["Normal", "Startup", "Surge_Zone", "Part_Load", "Equipment"],
}


# ─────────────────────────────────────────────
# LSTM AUTOENCODER — BUILD
# ─────────────────────────────────────────────


def build_bilstm_autoencoder(
    n_features: int, latent_units: int = 8, dropout_rate: float = 0.2
):
    """
    Build BiLSTM Autoencoder architecture.
    Encoder: BiLSTM(32) → BiLSTM(latent_units)
    Decoder: BiLSTM(latent_units) → BiLSTM(32) → TimeDistributed Dense

    Parameters
    ----------
    n_features : int
        Number of input features
    latent_units : int
        Size of latent space (bottleneck)
    dropout_rate : float
        Dropout for regularization
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input,
            LSTM,
            Bidirectional,
            RepeatVector,
            TimeDistributed,
            Dense,
            Dropout,
        )

        inp = Input(shape=(SEQUENCE_LENGTH, n_features))

        # ── Encoder ──
        x = Bidirectional(LSTM(32, return_sequences=True))(inp)
        x = Dropout(dropout_rate)(x)
        encoded = Bidirectional(LSTM(latent_units, return_sequences=False))(x)

        # ── Bottleneck → Repeat ──
        repeated = RepeatVector(SEQUENCE_LENGTH)(encoded)

        # ── Decoder ──
        x = Bidirectional(LSTM(latent_units, return_sequences=True))(repeated)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        decoded = TimeDistributed(Dense(n_features))(x)

        model = Model(inp, decoded, name="BiLSTM_Autoencoder")
        model.compile(optimizer="adam", loss="mae")

        return model

    except ImportError:
        print("[ML] TensorFlow not installed. Using placeholder model.")
        return None


# ─────────────────────────────────────────────
# SEQUENCE PREPARATION
# ─────────────────────────────────────────────


def create_sequences(data: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    """Convert 2D array to 3D sequences for LSTM input."""
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i : i + seq_len])
    return np.array(sequences)


def scale_features(df: pd.DataFrame, feature_cols: list, scaler_path: str = None):
    """Fit or load MinMaxScaler and return scaled data."""
    scaler = MinMaxScaler()

    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        scaled = scaler.transform(df[feature_cols].values)
        print(f"[ML] Scaler loaded from {scaler_path}")
    else:
        scaled = scaler.fit_transform(df[feature_cols].values)
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"[ML] Scaler saved to {scaler_path}")

    return scaled, scaler


# ─────────────────────────────────────────────
# TRAIN AUTOENCODER
# ─────────────────────────────────────────────


def train_autoencoder(
    df_normal: pd.DataFrame,
    feature_cols: list,
    model_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
):
    """
    Train BiLSTM Autoencoder on normal operation data.
    Saves model and threshold to model_dir.
    """
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Scale
    scaled, _ = scale_features(df_normal, feature_cols, scaler_path)

    # Sequences
    X = create_sequences(scaled)
    print(f"[ML] Training sequences shape: {X.shape}")

    # Build model
    model = build_bilstm_autoencoder(n_features=len(feature_cols))
    if model is None:
        print("[ML] Cannot train without TensorFlow.")
        return None, None

    # Train
    history = model.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    )

    # Save model
    model_path = os.path.join(model_dir, "autoencoder.keras")
    model.save(model_path)
    print(f"[ML] Model saved: {model_path}")

    # Calculate threshold: μ + 2.33σ (99th percentile, Wei et al. 2022)
    X_pred = model.predict(X, verbose=0)
    mae_scores = np.mean(np.abs(X_pred - X), axis=(1, 2))
    threshold = float(np.mean(mae_scores) + 2.33 * np.std(mae_scores))

    threshold_path = os.path.join(model_dir, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "mae_mean": float(np.mean(mae_scores)),
                "mae_std": float(np.std(mae_scores)),
            },
            f,
            indent=2,
        )

    print(f"[ML] Anomaly threshold: {threshold:.4f}")
    return model, threshold


# ─────────────────────────────────────────────
# PREDICT / DETECT ANOMALY
# ─────────────────────────────────────────────


def detect_anomaly(df: pd.DataFrame, feature_cols: list, model_dir: str):
    """
    Run anomaly detection on new data.
    Returns df with columns: mae, is_anomaly
    """
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path = os.path.join(model_dir, "autoencoder.keras")
    threshold_path = os.path.join(model_dir, "threshold.json")

    # Load threshold
    if not os.path.exists(threshold_path):
        print("[ML] No threshold found. Run training first.")
        df["mae"] = 0.0
        df["is_anomaly"] = False
        return df

    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    # Load scaler & scale
    scaled, _ = scale_features(df, feature_cols, scaler_path)
    X = create_sequences(scaled)

    # Load model and predict
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        X_pred = model.predict(X, verbose=0)
        mae_scores = np.mean(np.abs(X_pred - X), axis=(1, 2))

        # Align with original df (sequences reduce length by SEQUENCE_LENGTH-1)
        pad = np.zeros(SEQUENCE_LENGTH - 1)
        mae_full = np.concatenate([pad, mae_scores])

        df = df.copy()
        df["mae"] = mae_full
        df["threshold"] = threshold
        df["is_anomaly"] = df["mae"] > threshold

    except Exception as e:
        print(f"[ML] Detection error: {e}")
        df["mae"] = 0.0
        df["threshold"] = threshold
        df["is_anomaly"] = False

    return df


# ─────────────────────────────────────────────
# XGBOOST CLASSIFIER
# ─────────────────────────────────────────────


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_dir: str,
    equipment_type: str = "pump",
):
    """
    Train XGBoost + SMOTE classifier for anomaly type classification.
    Target F1-score > 0.85.
    """
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    os.makedirs(model_dir, exist_ok=True)

    # SMOTE oversampling for class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[ML] After SMOTE: {X_resampled.shape[0]} samples")

    # XGBoost
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    clf.fit(X_resampled, y_resampled)

    # Save
    clf_path = os.path.join(model_dir, f"xgboost_{equipment_type}.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"[ML] Classifier saved: {clf_path}")
    return clf


def classify_anomaly(
    features: np.ndarray, model_dir: str, equipment_type: str = "pump"
):
    """
    Predict anomaly class using trained XGBoost classifier.
    Returns (class_label, confidence)
    """
    clf_path = os.path.join(model_dir, f"xgboost_{equipment_type}.pkl")

    if not os.path.exists(clf_path):
        return "Unknown", 0.0

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    proba = clf.predict_proba(features.reshape(1, -1))[0]
    class_idx = np.argmax(proba)
    classes = ANOMALY_CLASSES.get(equipment_type, ["Normal", "Anomaly"])
    label = classes[class_idx] if class_idx < len(classes) else "Unknown"

    return label, float(proba[class_idx])
