# train_engine.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier

os.makedirs("saved_models", exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Synthetic Financial Transactions Generation ---
print("1. Generating synthetic transaction records...")
n_samples = 5000

amounts = np.random.exponential(scale=50, size=n_samples) + 5
hour_of_day = np.random.randint(0, 24, size=n_samples)
distance_from_home_km = np.random.exponential(scale=10, size=n_samples)
transaction_velocity = np.random.poisson(lam=2, size=n_samples)

# Labels: Normal (0), Fraud (1)
fraud_indicator = (
    (amounts > 250).astype(int) * 0.4
    + (distance_from_home_km > 50).astype(int) * 0.4
    + (transaction_velocity > 5).astype(int) * 0.3
)
y = (fraud_indicator + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

X = pd.DataFrame({
    "amount": amounts,
    "hour": hour_of_day,
    "distance_km": distance_from_home_km,
    "velocity_last_24h": transaction_velocity,
})

# Preprocessing Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "saved_models/scaler.joblib")

# --- 2. Train ML Model (XGBoost Classifier) ---
print("2. Training XGBoost Risk Model (Supervised ML)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
xgb_model = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42
)
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, "saved_models/xgb_fraud_model.joblib")
print("ML Model saved: saved_models/xgb_fraud_model.joblib")


# --- 3. Train DL Model (PyTorch Autoencoder - Unsupervised Anomaly) ---
print("3. Training PyTorch Autoencoder (Deep Learning Anomaly Detector)...")


class TransactionAutoencoder(nn.Module):

  def __init__(self, input_dim=4):
    super(TransactionAutoencoder, self).__init__()
    # Encoder
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, 8),
        nn.ReLU(),
        nn.Linear(8, 2),  # Compressed bottleneck
    )
    # Decoder
    self.decoder = nn.Sequential(
        nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, input_dim)
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


# Autoencoders train on normal transactions (y == 0)
X_normal = X_scaled[y == 0]
tensor_normal = torch.tensor(X_normal, dtype=torch.float32)

autoencoder = TransactionAutoencoder(input_dim=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# Training loop
for epoch in range(15):
  optimizer.zero_grad()
  outputs = autoencoder(tensor_normal)
  loss = criterion(outputs, tensor_normal)
  loss.backward()
  optimizer.step()

torch.save(autoencoder.state_dict(), "saved_models/autoencoder_model.pth")
print("DL Model saved: saved_models/autoencoder_model.pth")
print("\nAll models trained and persisted successfully!")