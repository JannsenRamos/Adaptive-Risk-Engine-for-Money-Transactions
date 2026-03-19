"""
ML Model wrapper for the SAVE Engine.
Trains a RandomForestClassifier on either:
  1. The PaySim Kaggle dataset  (data/PS_20174392719_1491204439457_log.csv)  ← preferred
  2. Synthetic simulation data generated internally                           ← fallback

Exposes a predict_proba interface for app.py.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── PaySim CSV path (relative to project root) ──────────────────────────────
_CSV_PATH = Path(__file__).parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"

# ── Transaction types → Purpose Drift proxy ─────────────────────────────────
# Fraud in PaySim only occurs in CASH_OUT and TRANSFER → higher drift scores.
_TYPE_DRIFT = {
    "PAYMENT":   0.05,
    "CASH_IN":   0.10,
    "DEBIT":     0.30,
    "TRANSFER":  0.75,
    "CASH_OUT":  0.85,
}

# Amount above which amount_risk saturates at 1.0 (99th pct of PaySim fraud)
_AMOUNT_CAP = 10_000_000.0

FEATURE_COLS = ["drift", "preamble_score", "vel_penalty", "amount_risk"]


# ── PaySim loader ────────────────────────────────────────────────────────────

def _load_paysim_data(
    n_normal: int = 2000,
    n_fraud: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads PaySim CSV and engineers the 4 SAVE features from raw columns.

    PaySim columns used:
        type          → drift  (transaction purpose proxy)
        amount        → amount_risk
        oldbalanceOrg → vel_penalty  (balance-depletion ratio)
        newbalanceOrig
        isFraud       → label + preamble_score proxy

    Returns a DataFrame with [drift, preamble_score, vel_penalty,
                               amount_risk, is_attack].
    """
    needed = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"]
    df = pd.read_csv(_CSV_PATH, usecols=needed, low_memory=False)

    rng = np.random.default_rng(random_state)

    # ── Feature engineering ─────────────────────────────────────────
    # 1. Drift — from transaction type
    df["drift"] = df["type"].map(_TYPE_DRIFT).fillna(0.5)

    # 2. Preamble score — fraud = flash transaction (no navigation), normal = proper nav
    #    For fraud rows we draw from [0.7, 0.9] (high anomaly).
    #    For normal rows we draw from [0.1, 0.7] weighted toward 0.1.
    n = len(df)
    preamble = np.where(
        df["isFraud"] == 1,
        rng.uniform(0.70, 0.90, n),
        rng.choice([0.1, 0.7], n, p=[0.90, 0.10]),
    )
    df["preamble_score"] = preamble

    # 3. Velocity penalty — how aggressively was the balance depleted?
    #    ratio = (old - new) / max(old, 1)  → [0, 1+]
    #    vel_penalty maps: 0 depletion → 1.0, full drain → 3.0
    bal_delta = (df["oldbalanceOrg"] - df["newbalanceOrig"]) / df["oldbalanceOrg"].clip(lower=1)
    df["vel_penalty"] = (1.0 + bal_delta.clip(0, 1) * 2.0).round(2)

    # 4. Amount risk — normalise against PaySim's 99th-percentile fraud amount
    df["amount_risk"] = (df["amount"] / _AMOUNT_CAP).clip(0, 1)

    df["is_attack"] = df["isFraud"].astype(bool)

    # ── Balanced sampling ───────────────────────────────────────────
    normal_pool = df[df["is_attack"] == False]
    fraud_pool  = df[df["is_attack"] == True]

    n_normal_avail = len(normal_pool)
    n_fraud_avail  = len(fraud_pool)

    n_normal = min(n_normal, n_normal_avail)
    # Oversample fraud with replacement if needed (only 8 213 rows in PaySim)
    replace_fraud = n_fraud > n_fraud_avail
    n_fraud = min(n_fraud, n_fraud_avail * 3)  # cap at 3× available

    normal_sample = normal_pool.sample(n=n_normal, random_state=random_state)
    fraud_sample  = fraud_pool.sample(n=n_fraud, replace=replace_fraud,
                                      random_state=random_state)

    out = pd.concat([normal_sample, fraud_sample], ignore_index=True).sample(
        frac=1, random_state=random_state
    )
    return out[FEATURE_COLS + ["is_attack"]]


# ── Synthetic fallback ───────────────────────────────────────────────────────

def _generate_training_data(
    n_normal: int = 2000,
    n_attack: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a synthetic labeled dataset without running the full SAVE pipeline.
    Used only when the PaySim CSV is not present.
    """
    rng = np.random.default_rng(random_state)

    normal = pd.DataFrame({
        "drift":          rng.uniform(0.0, 0.4, n_normal),
        "preamble_score": rng.choice([0.1, 0.7], n_normal, p=[0.85, 0.15]),
        "vel_penalty":    rng.choice([1.0, 2.0], n_normal, p=[0.90, 0.10]),
        "amount_risk":    rng.uniform(0.0, 0.3, n_normal),
        "is_attack":      False,
    })

    attack = pd.DataFrame({
        "drift":          rng.uniform(0.5, 1.0, n_attack),
        "preamble_score": rng.choice([0.9, 0.7], n_attack, p=[0.80, 0.20]),
        "vel_penalty":    rng.choice([2.0, 3.0], n_attack, p=[0.60, 0.40]),
        "amount_risk":    rng.uniform(0.5, 1.0, n_attack),
        "is_attack":      True,
    })

    return pd.concat([normal, attack], ignore_index=True).sample(
        frac=1, random_state=random_state
    )


# ── Model class ──────────────────────────────────────────────────────────────

class SAVEModel:
    """
    Thin wrapper around a sklearn RandomForest trained on PaySim (or synthetic)
    SAVE-signal data.
    """

    def __init__(self):
        self._model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        )
        self._scaler   = StandardScaler()
        self._trained  = False
        self.data_source   = "unknown"
        self.training_rows = 0

    def train(self):
        if _CSV_PATH.exists():
            try:
                df = _load_paysim_data()
                self.data_source = "PaySim (Kaggle)"
            except Exception as exc:
                warnings.warn(f"PaySim load failed ({exc}), falling back to synthetic.")
                df = _generate_training_data()
                self.data_source = "Synthetic (fallback)"
        else:
            df = _generate_training_data()
            self.data_source = "Synthetic (no CSV found)"

        self.training_rows = len(df)
        X = df[FEATURE_COLS].values
        y = df["is_attack"].astype(int).values

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._trained = True

    def predict_proba(self, drift: float, preamble_score: float,
                      vel_penalty: float, amount_risk: float) -> float:
        """Returns P(attack) in [0, 1]."""
        if not self._trained:
            self.train()
        X = np.array([[drift, preamble_score, vel_penalty, amount_risk]])
        X_scaled = self._scaler.transform(X)
        return float(self._model.predict_proba(X_scaled)[0][1])

    def feature_importances(self) -> dict:
        if not self._trained:
            self.train()
        return dict(zip(FEATURE_COLS, self._model.feature_importances_.tolist()))
