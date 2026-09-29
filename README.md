# 🛡️ SentinelPay | Enterprise Fraud, Anomaly & RAG Intelligence Engine

A production-grade FinTech intelligence platform engineered to detect payment anomalies, classify transaction fraud, and automate regulatory compliance reporting.

## 🚀 Key Architectural Pillars

- **Hybrid AI Detection Core:**
  - **Supervised Machine Learning:** XGBoost Classifier for probability-based pattern recognition.
  - **Deep Learning Anomaly Detection:** PyTorch Deep Autoencoder computing reconstruction loss (MSE) against baseline financial behaviors.
- **Regulatory Compliance RAG (Retrieval-Augmented Generation):**
  - Contextual retrieval across FinTech guidelines (AML / RBI thresholds).
  - Automated generation of Forensic Suspicious Activity Reports (SAR).
- **Asynchronous API Gateway:** Built on FastAPI with strict Pydantic payload schema validation.
- **Enterprise Ledger:** SQLAlchemy ORM coupled with SQLite/PostgreSQL for immutable audit logging.
- **Interactive Operations Center:** Streamlit dashboard for real-time risk scoring, CSV bulk stream ingestion, and audit analysis.

## 🛠️ Tech Stack

- **Languages & Frameworks:** Python 3.13, FastAPI, Streamlit, SQLAlchemy
- **Machine Learning & AI:** PyTorch, XGBoost, Scikit-Learn, NumPy, Pandas
- **Persistence & Formats:** SQLite, CSV Ingestion, Joblib

## ⚙️ Quickstart

1. **Clone & Install Dependencies:**
   ```bash
   pip install -r requirements.txt


### 1. Train AI Engine
```bash
python train_engine.py


### 1. Start API Gateway
```bash
uvicorn main:app --reload
```

### 2. Launch Dashboard
```bash
streamlit run dashboard.py
```