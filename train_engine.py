# train_engine.py

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim

from xgboost import XGBClassifier


# ============================================================
# SENTINELPAY - MODEL TRAINING ENGINE
# ============================================================

print("=" * 60)
print(" SENTINELPAY - MODEL TRAINING ENGINE")
print("=" * 60)


# ------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------

os.makedirs("saved_models", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

N_SAMPLES = 5000
TARGET_FRAUD_RATE = 0.05

print("\n1. Generating synthetic transaction records...")


# ------------------------------------------------------------
# 1. Generate Synthetic Transactions
# ------------------------------------------------------------

amounts = np.random.exponential(
    scale=50,
    size=N_SAMPLES
) + 5

hour_of_day = np.random.randint(
    0,
    24,
    size=N_SAMPLES
)

distance_from_home_km = np.random.exponential(
    scale=10,
    size=N_SAMPLES
)

transaction_velocity = np.random.poisson(
    lam=2,
    size=N_SAMPLES
)


# ------------------------------------------------------------
# 2. Create Synthetic Risk Score
# ------------------------------------------------------------

risk_score = (
    (amounts > 250).astype(int) * 0.4
    + (distance_from_home_km > 50).astype(int) * 0.4
    + (transaction_velocity > 5).astype(int) * 0.3
)

# Small random variation
risk_score += np.random.normal(
    0,
    0.10,
    N_SAMPLES
)


# ------------------------------------------------------------
# 3. Create Balanced-Enough Synthetic Fraud Labels
# ------------------------------------------------------------

# Instead of using a fixed 0.5 threshold,
# select approximately the top 5% highest-risk transactions.

fraud_threshold = np.percentile(
    risk_score,
    100 * (1 - TARGET_FRAUD_RATE)
)

y = (
    risk_score >= fraud_threshold
).astype(int)


# ------------------------------------------------------------
# 4. Create Feature DataFrame
# ------------------------------------------------------------

X = pd.DataFrame({
    "amount": amounts,
    "hour": hour_of_day,
    "distance_km": distance_from_home_km,
    "velocity_last_24h": transaction_velocity,
})


print(f"Total transactions : {len(X)}")
print(f"Normal transactions: {(y == 0).sum()}")
print(f"Fraud transactions : {(y == 1).sum()}")
print(
    f"Fraud rate         : {y.mean() * 100:.2f}%"
)


# ------------------------------------------------------------
# 5. Train / Validation / Test Split
# ------------------------------------------------------------

print("\n2. Splitting dataset...")

# First split: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Second split:
# 15% validation
# 15% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Testing samples    : {len(X_test)}")


# ------------------------------------------------------------
# 6. StandardScaler
# ------------------------------------------------------------

print("\n3. Fitting StandardScaler on training data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


joblib.dump(
    scaler,
    "saved_models/scaler.joblib"
)

print(
    "Scaler saved: saved_models/scaler.joblib"
)


# ------------------------------------------------------------
# 7. XGBoost Fraud Detection Model
# ------------------------------------------------------------

print("\n4. Training XGBoost Fraud Detection Model...")


# Handles class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)


print(
    f"Scale Pos Weight: {scale_pos_weight:.2f}"
)


xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)


xgb_model.fit(
    X_train_scaled,
    y_train
)


joblib.dump(
    xgb_model,
    "saved_models/xgb_fraud_model.joblib"
)

print(
    "XGBoost model saved: "
    "saved_models/xgb_fraud_model.joblib"
)


# ------------------------------------------------------------
# 8. Validation Predictions
# ------------------------------------------------------------

print("\n5. Finding fraud classification threshold...")


val_probabilities = xgb_model.predict_proba(
    X_val_scaled
)[:, 1]


# Try thresholds from 0.10 to 0.90
thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


best_threshold = 0.50
best_f1 = -1


for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = float(threshold)


print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1 Score       : "
    f"{best_f1:.4f}"
)


# ------------------------------------------------------------
# 9. Final Test Evaluation
# ------------------------------------------------------------

print("\n6. Evaluating XGBoost model...")


test_probabilities = xgb_model.predict_proba(
    X_test_scaled
)[:, 1]


test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


print("\nXGBoost Performance")
print("-" * 40)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 10. PyTorch Autoencoder
# ------------------------------------------------------------

print("\n7. Training PyTorch Autoencoder...")


class TransactionAutoencoder(nn.Module):

    def __init__(self, input_dim=4):

        super(TransactionAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# Only NORMAL training transactions
X_train_normal = X_train_scaled[
    y_train == 0
]


tensor_normal = torch.tensor(
    X_train_normal,
    dtype=torch.float32
)


autoencoder = TransactionAutoencoder(
    input_dim=4
)


criterion = nn.MSELoss()


optimizer = optim.Adam(
    autoencoder.parameters(),
    lr=0.01
)


EPOCHS = 30


print(
    f"Training Autoencoder for "
    f"{EPOCHS} epochs..."
)


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    outputs = autoencoder(
        tensor_normal
    )

    loss = criterion(
        outputs,
        tensor_normal
    )

    loss.backward()

    optimizer.step()


    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {loss.item():.6f}"
        )


# ------------------------------------------------------------
# 11. Autoencoder Anomaly Threshold
# ------------------------------------------------------------

print("\n8. Calculating Autoencoder anomaly scores...")


# Validation normal transactions
X_val_normal = X_val_scaled[
    y_val == 0
]


tensor_val_normal = torch.tensor(
    X_val_normal,
    dtype=torch.float32
)


with torch.no_grad():

    reconstructed = autoencoder(
        tensor_val_normal
    )

    reconstruction_errors = torch.mean(
        (reconstructed - tensor_val_normal) ** 2,
        dim=1
    ).numpy()


# 95th percentile threshold
anomaly_threshold = float(
    np.percentile(
        reconstruction_errors,
        95
    )
)


print(
    f"Autoencoder anomaly threshold: "
    f"{anomaly_threshold:.6f}"
)


torch.save(
    autoencoder.state_dict(),
    "saved_models/autoencoder_model.pth"
)


print(
    "Autoencoder saved: "
    "saved_models/autoencoder_model.pth"
)


# ------------------------------------------------------------
# 12. Save Metrics
# ------------------------------------------------------------

metrics = {

    "dataset": {
        "total_samples": int(len(X)),
        "normal_samples": int((y == 0).sum()),
        "fraud_samples": int((y == 1).sum()),
        "fraud_rate": float(y.mean()),
    },

    "xgboost": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "classification_threshold": float(
            best_threshold
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "confusion_matrix": cm.tolist(),
    },

    "autoencoder": {
        "anomaly_threshold": float(
            anomaly_threshold
        )
    }
}


with open(
    "saved_models/model_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print(
    "\nMetrics saved: "
    "saved_models/model_metrics.json"
)


# ------------------------------------------------------------
# 13. Final Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)

print(
    " MODEL TRAINING COMPLETED SUCCESSFULLY"
)

print("=" * 60)

print("\nSaved files:")

print(
    "1. saved_models/scaler.joblib"
)

print(
    "2. saved_models/xgb_fraud_model.joblib"
)

print(
    "3. saved_models/autoencoder_model.pth"
)

print(
    "4. saved_models/model_metrics.json"
)

print(
    "\nSentinelPay training pipeline is ready."
)