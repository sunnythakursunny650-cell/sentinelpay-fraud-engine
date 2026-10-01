# 🛡️ SentinelPay | Hybrid Fraud & Anomaly Risk Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

SentinelPay is a **hybrid machine-learning and deep-learning transaction-risk demonstration system** designed to evaluate synthetic transaction data, combine supervised fraud-probability and anomaly signals, retrieve matching internal demo-policy sections, generate a compliance review report, and persist evaluation results in a SQLite audit ledger.

> **Important:** SentinelPay is a demonstration project. The models are trained and evaluated on **synthetic transaction data**, and the policy rules are illustrative internal demo rules. They are not official RBI/AML regulations, legal advice, regulatory guidance, or actual compliance controls.

---

## 👨‍💻 Author & Project Profile

- **Developer:** **Sunny Thakur**
- **Focus:** Python, Machine Learning & Risk Analytics
- **GitHub:** [sunnythakursunny650-cell](https://github.com/sunnythakursunny650-cell)
- **LinkedIn:** [Sunny Thakur](https://www.linkedin.com/in/sunny-thakur-4a56103b9/)
- **Dashboard:** Streamlit

---

## 🚀 Project Overview

SentinelPay combines two independent model signals:

### 1. Supervised Fraud Detection

An **XGBoost classifier** estimates a fraud probability from four transaction features:

- Transaction amount
- Transaction hour
- Distance from configured baseline location
- Transaction velocity during the previous 24 hours

The training pipeline creates synthetic transaction records, splits them into training, validation, and test sets, fits the scaler only on training data, trains XGBoost, and selects a classification threshold using validation F1 score.

### 2. Deep Learning Anomaly Detection

A **PyTorch Autoencoder** learns reconstruction behavior from normal synthetic transactions.

Architecture:

```text
4 → 8 → 2 → 8 → 4
```

The reconstruction Mean Squared Error (MSE) is used as an anomaly signal.

### 3. Hybrid Decision Engine

The API combines the XGBoost probability and Autoencoder reconstruction loss.

Current configured thresholds:

| Signal | Threshold |
| :--- | :---: |
| XGBoost fraud probability | `0.70` |
| Autoencoder anomaly loss | `1.3337` |
| Flagged-review ML boundary | `0.35` |
| Flagged-review DL boundary | `0.8669` |

Decision logic:

| Condition | Decision |
| :--- | :--- |
| `ML Prob >= 0.70` OR `DL Loss >= 1.3337` | **DECLINED** |
| `ML Prob >= 0.35` OR `DL Loss >= 0.8669` | **FLAGGED_REVIEW** |
| Otherwise | **APPROVED** |

---

## 🧠 Policy Retrieval & Compliance Review

SentinelPay includes a lightweight **rule-based policy retrieval layer**.

It reads sections from:

```text
compliance_policy.txt
```

The current demo policy contains illustrative rules for:

- High transaction velocity
- Geographic deviation
- Unusual transaction size
- Baseline transaction activity

When a transaction matches a configured rule, the relevant policy section is returned as a citation.

The system then generates a structured **Compliance Review Report** containing:

- Account information
- Transaction details
- Model risk signals
- Matched policy sections
- Recommended internal review action
- System disclaimer

### What it does NOT do

SentinelPay does **not**:

- Submit SARs
- Submit regulatory filings
- Freeze accounts
- Perform real bank authorization
- Make legal determinations
- Connect to RBI or other regulatory systems
- Claim that its demo rules are official AML/RBI requirements

The policy engine is **rule-based retrieval + templated report generation**. It is not an LLM-based generative RAG system.

---

## 🏗️ Architecture

```text
                 ┌─────────────────────┐
                 │   Streamlit UI      │
                 │ Live / Batch /      │
                 │ Ledger / Blueprint  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     FastAPI API     │
                 │ Pydantic Validation │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌────────────────┐    ┌─────────────────┐
        │    XGBoost     │    │ PyTorch         │
        │ Fraud Signal   │    │ Autoencoder     │
        └───────┬────────┘    └────────┬────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                 ┌─────────────────────┐
                 │ Hybrid Decision     │
                 │ APPROVED / FLAGGED  │
                 │ / DECLINED          │
                 └──────────┬──────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
       ┌──────────────────┐     ┌──────────────────┐
       │ Demo Policy      │     │ SQLite Audit     │
       │ Retrieval        │     │ Ledger           │
       └────────┬─────────┘     └──────────────────┘
                ▼
       Compliance Review
             Report
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Supervised ML** | XGBoost, Scikit-learn |
| **Deep Learning** | PyTorch Autoencoder |
| **Data Processing** | Pandas, NumPy |
| **Policy Retrieval** | Python rule-based section matching |
| **Database** | SQLite, SQLAlchemy |
| **Dashboard** | Streamlit |
| **Model Persistence** | Joblib, PyTorch model weights |
| **Containerization** | Docker, Docker Compose |

---

## 📊 Model Training Results

The current training run used **5,000 synthetic transactions**:

```text
Total transactions : 5000
Normal transactions: 4750
Fraud transactions : 250
Synthetic fraud rate: 5.00%
```

### XGBoost Test Results

| Metric | Result |
| :--- | :---: |
| Accuracy | **96.80%** |
| Precision | **74.07%** |
| Recall | **54.05%** |
| F1 Score | **62.50%** |
| ROC-AUC | **80.08%** |

### Test Confusion Matrix

```text
[[706   7]
 [ 17  20]]
```

### XGBoost Threshold Selection

```text
Best validation threshold: 0.70
Validation F1 Score: 0.6269
```

### Autoencoder Threshold

```text
Anomaly threshold: 1.333665
```

> These metrics describe performance on the project's **synthetic test data only**. They should not be interpreted as real-world banking fraud performance.

---

## 📡 API Endpoints

### Health Check

```text
GET /
```

Returns:

- API status
- API version
- Model engine
- Configured ML threshold
- Configured DL threshold

---

### Evaluate Transaction

```text
POST /api/v1/evaluate-transaction
```

Evaluates one transaction and returns:

- Transaction ID
- Account ID
- Risk decision
- XGBoost fraud probability
- Autoencoder anomaly score
- Configured thresholds
- Matched policy sections
- Compliance review report

Example request:

```json
{
  "account_id": "ACC-TEST-003",
  "amount": 450,
  "hour_of_day": 3,
  "distance_from_home_km": 85.5,
  "velocity_last_24h": 6
}
```

---

### Batch CSV Evaluation

```text
POST /api/v1/batch-evaluate-csv
```

Processes multiple transactions from a CSV file.

Required columns:

```text
account_id
amount
hour_of_day
distance_from_home_km
velocity_last_24h
```

The endpoint returns:

- Total processed records
- Approved count
- Flagged-review count
- Declined count
- Configured thresholds
- Sample processed records

---

### Audit Ledger

```text
GET /api/v1/ledger
```

Returns persisted transaction-risk records.

Supports filtering by:

- Account ID
- Decision status

The dashboard also provides a CSV export option for the audit ledger.

---

## 🖥️ Streamlit Dashboard

The Streamlit dashboard contains four main sections.

### ⚡ Live Risk Simulation

Submit a transaction and view:

- XGBoost fraud probability
- Autoencoder anomaly score
- Final risk decision
- Matched demo-policy sections
- Compliance review report

---

### 📁 Batch CSV Processing

Upload transaction CSV data and process multiple records through the same API decision engine.

Example workflow:

```text
CSV Upload
    ↓
FastAPI
    ↓
XGBoost + Autoencoder
    ↓
Risk Decision
    ↓
SQLite Audit Ledger
```

---

### 📋 Audit Ledger

View stored transaction evaluations including:

- Transaction ID
- Account ID
- Amount
- ML probability
- DL anomaly score
- Decision
- Reason
- Timestamp

The dashboard also provides:

```text
Export Ledger Snapshot (CSV)
```

---

### 🔍 System Architecture

The architecture section displays:

- FastAPI backend
- XGBoost model
- PyTorch Autoencoder
- Rule-based policy retrieval
- SQLite persistence
- Streamlit dashboard
- Current decision thresholds
- Project limitations

---

## ⚙️ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/sunnythakursunny650-cell/sentinelpay-fraud-engine.git
cd sentinelpay-fraud-engine
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Models

```bash
python train_engine.py
```

This creates:

```text
saved_models/
├── scaler.joblib
├── xgb_fraud_model.joblib
├── autoencoder_model.pth
└── model_metrics.json
```

### 4. Start the FastAPI Server

```bash
uvicorn main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

### 5. Start the Streamlit Dashboard

Open a second terminal:

```bash
streamlit run dashboard.py
```

---

## 🐳 Docker

The project also includes Docker configuration.

Run:

```bash
docker compose up --build
```

> For Docker deployment, the dashboard/API service URL should be configured for the container network rather than assuming `127.0.0.1` refers to the API container.

---

## 📁 Project Structure

```text
sentinelpay-fraud-engine/
│
├── main.py
├── train_engine.py
├── dashboard.py
├── database.py
├── rag_engine.py
├── compliance_policy.txt
├── generate_csv.py
├── transactions.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
│
└── saved_models/
    ├── scaler.joblib
    ├── xgb_fraud_model.joblib
    ├── autoencoder_model.pth
    └── model_metrics.json
```

---

## 🔐 Project Limitations

This project is intentionally a **software and ML demonstration**, not a production financial system.

Important limitations:

1. Transaction data is synthetic.
2. Fraud labels are generated synthetically rather than obtained from real banking data.
3. Model performance is therefore not representative of real-world fraud detection.
4. Policy sections are illustrative internal demo rules.
5. Policy retrieval is rule-based rather than LLM-based RAG.
6. Compliance reports are generated for review and do not constitute legal or regulatory filings.
7. SQLite is used for demonstration persistence.
8. Real financial deployments would require stronger security, authentication, authorization, monitoring, data governance, model validation, and regulatory/compliance controls.

---

## 🎯 Learning & Engineering Goals

SentinelPay demonstrates an end-to-end workflow combining:

- Synthetic data generation
- Data preprocessing
- Supervised machine learning
- Deep-learning anomaly detection
- Threshold selection
- Model evaluation
- FastAPI inference APIs
- Pydantic validation
- Rule-based policy retrieval
- Compliance-style report generation
- SQLite persistence
- Batch CSV processing
- Streamlit dashboard development
- Docker-based deployment structure

---

## 🧪 Demonstrated Test Results

The current project has been tested locally across the major dashboard workflows.

### Live Transaction Evaluation

Successfully tested:

```text
Decision: DECLINED
XGBoost Fraud Probability: 99.4%
Autoencoder Anomaly Score: 43.6719
```

Multiple demo-policy sections were matched successfully.

### Batch Processing

Successfully processed:

```text
Total Processed: 200
Approved: 107
Flagged Review: 45
Declined: 48
```

The counts are consistent:

```text
107 + 45 + 48 = 200
```

### Audit Ledger

The SQLite audit ledger successfully displayed:

- Transaction records
- ML probabilities
- DL anomaly scores
- Decisions
- Reasons
- Timestamps
- Account filtering
- CSV export functionality

### System Architecture

The dashboard successfully displays the current:

```text
XGBoost Threshold: 0.70
Autoencoder Threshold: 1.3337
```

and the synthetic-data/project limitation disclaimer.

---

## 🔎 Policy Rules

The project uses an illustrative internal policy file:

```text
compliance_policy.txt
```

Example policy categories include:

```text
SECTION-101
High-Velocity Anomaly Rule

SECTION-204
Geographic Location Deviation Rule

SECTION-309
Unusual Ticket Size Rule

SECTION-412
Baseline Clearance Rule
```

These sections are used only for demonstration-level policy retrieval and report generation.

> They should not be interpreted as official regulatory requirements.

---

## 📌 Why Hybrid ML + DL?

The project demonstrates two different approaches to transaction-risk analysis.

### XGBoost

Provides a supervised probability signal based on labeled synthetic examples.

```text
Transaction Features
        ↓
     XGBoost
        ↓
Fraud Probability
```

### Autoencoder

Provides an anomaly signal based on reconstruction error.

```text
Transaction Features
        ↓
   Autoencoder
        ↓
Reconstruction Error
```

### Combined

```text
XGBoost Probability
        +
Autoencoder Anomaly Score
        ↓
Hybrid Decision Engine
        ↓
APPROVED
FLAGGED_REVIEW
DECLINED
```

This architecture demonstrates how different model signals can be combined within a single API-based transaction-risk workflow.

---

## 📚 Technical Concepts Demonstrated

This project covers practical concepts across Python, machine learning, deep learning, APIs, databases, and deployment.

### Python

- Functions
- Classes
- Exception handling
- File handling
- JSON operations
- Data processing
- Modular project structure

### Machine Learning

- Synthetic data generation
- Feature preprocessing
- StandardScaler
- XGBoost
- Train/validation/test split
- Classification threshold selection
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Deep Learning

- PyTorch
- Neural networks
- Autoencoder architecture
- Reconstruction loss
- Anomaly thresholding

### Backend Development

- FastAPI
- REST APIs
- Pydantic validation
- API endpoints
- File uploads
- Batch processing

### Database

- SQLite
- SQLAlchemy
- ORM models
- Persistent transaction records
- Filtering
- Audit-style logging

### Dashboard

- Streamlit
- Interactive forms
- Data tables
- CSV upload
- CSV export
- Risk visualization

### Deployment

- Docker
- Docker Compose
- Uvicorn
- Model persistence

---

## 🚀 Future Improvements

Possible future improvements include:

- Real-world anonymized transaction datasets
- Stronger feature engineering
- Cross-validation
- Model calibration
- Explainable AI techniques
- SHAP-based feature explanations
- Authentication and authorization
- API rate limiting
- Structured application logging
- PostgreSQL for larger deployments
- Model monitoring
- Data drift detection
- Automated model retraining pipelines
- Proper external compliance knowledge sources
- LLM-based RAG with verified source documents
- Production-grade security and observability

---

## ⚠️ Disclaimer

SentinelPay is an educational and portfolio-oriented demonstration project.

It is **not**:

- A banking system
- A payment authorization system
- A fraud investigation platform
- A regulatory reporting system
- A legal compliance system
- An official RBI/AML implementation
- A replacement for professional financial, legal, or compliance controls

All transaction data used for model training and evaluation is synthetic.

All policy rules included in the project are illustrative demonstration rules.

---

## 👨‍💻 Developer

### Sunny Thakur

**Python | Machine Learning | Data & Risk Analytics**

GitHub:

https://github.com/sunnythakursunny650-cell

LinkedIn:

https://www.linkedin.com/in/sunny-thakur-4a56103b9/

---

## ⭐ Project Summary

SentinelPay demonstrates an end-to-end transaction-risk workflow combining:

```text
Synthetic Data
      ↓
Data Preprocessing
      ↓
XGBoost Fraud Detection
      +
PyTorch Autoencoder
      ↓
Hybrid Risk Decision
      ↓
Rule-Based Policy Retrieval
      ↓
Compliance Review Report
      ↓
SQLite Audit Ledger
      ↓
Streamlit Operations Dashboard
```

The project focuses on demonstrating practical integration of **Machine Learning, Deep Learning, FastAPI, Streamlit, SQLAlchemy, SQLite, Docker, and Python-based policy retrieval** in a single portfolio project.

---

© 2026 Sunny Thakur — SentinelPay demonstration project.