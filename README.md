# 🛡️ SentinelPay | Enterprise Fraud, Anomaly & RAG Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

A production-grade FinTech intelligence platform engineered to detect payment anomalies, classify transaction fraud, and automate regulatory compliance reporting via dynamic Retrieval-Augmented Generation (RAG).

---

## 👨‍💻 Author & Engineering Profile

* **Developer:** **Sunny Thakur**
* **Role:** AI & Machine Learning Infrastructure Developer
* **GitHub:** [![GitHub](https://img.shields.io/badge/GitHub-sunnythakursunny650--cell-181717?style=flat-square&logo=github)](https://github.com/sunnythakursunny650-cell)
* **LinkedIn:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-Sunny_Thakur-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/sunny-thakur-4a56103b9/)
* **Operations Dashboard:** Live UI built with [Streamlit](https://streamlit.io/)

---

## 🚀 Key Architectural Pillars

- **Hybrid AI Detection Core:**
  - **Supervised Machine Learning:** XGBoost Classifier for probability-based pattern recognition.
  - **Deep Learning Anomaly Detection:** PyTorch Deep Autoencoder computing reconstruction loss (MSE) against baseline financial behaviors.
- **Regulatory Compliance RAG (Retrieval-Augmented Generation):**
  - Contextual retrieval across FinTech guidelines (AML / RBI thresholds).
  - Automated generation of Forensic Suspicious Activity Reports (SAR).
- **Asynchronous API Gateway:** Built on FastAPI with strict Pydantic payload schema validation.
- **Enterprise Ledger:** SQLAlchemy ORM coupled with SQLite for immutable audit logging.
- **Interactive Operations Center:** Streamlit dashboard for real-time risk scoring, CSV bulk stream ingestion, and audit analysis.
- **Containerized Orchestration:** Multi-service deployment powered by Docker and Docker Compose.

---

## 🛠️ Tech Stack & Dependencies

| Layer | Technologies |
| :--- | :--- |
| **Backend & REST API** | FastAPI, Uvicorn, Pydantic |
| **Machine Learning (Supervised)** | XGBoost, Scikit-Learn |
| **Deep Learning (Unsupervised)** | PyTorch (Autoencoder Architecture) |
| **GenAI / Compliance** | RAG Engine, AML Policy Knowledge Base |
| **Database & ORM** | SQLite, SQLAlchemy |
| **Operations UI** | Streamlit, Pandas, NumPy |
| **DevOps & Containers** | Docker, Docker Compose |

---

## ⚙️ Quickstart & Execution

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/sunnythakursunny650-cell/sentinelpay-fraud-engine.git](https://github.com/sunnythakursunny650-cell/sentinelpay-fraud-engine.git)
cd sentinelpay-fraud-engine
pip install -r requirements.txt
```

### 2. Train AI Engine
```bash
python train_engine.py
```

### 3. Start API Gateway
```bash
uvicorn main:app --reload
```

### 4. Launch Dashboard
```bash
streamlit run dashboard.py
```

### 5. Run via Docker Compose (Optional)
```bash
docker compose up --build
```

---

## ⚖️ Compliance & SAR Architecture

Transactions evaluated as **DECLINED** or **FLAGGED_REVIEW** automatically trigger the internal RAG synthesizer:
1. Cross-references parameters against `compliance_policy.txt` (Velocity, Geo-cluster, Ticket limits).
2. Attaches exact regulatory breach citations (e.g., `[SECTION-101]`, `[SECTION-204]`).
3. Generates an automated legal forensic dossier ready for regulatory filing.

---

© 2026 Sunny Thakur. Built for Enterprise FinTech AI Infrastructure.