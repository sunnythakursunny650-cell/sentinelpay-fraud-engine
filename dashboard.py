# dashboard.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="SentinelPay | Enterprise Fraud & Anomaly Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Enterprise Theme
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 16px;
        border-radius: 8px;
    }
    .author-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #1e293b;
        color: #38bdf8;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: System Metadata & Author Profile ---
with st.sidebar:
    st.markdown('<div class="author-badge">LEAD AI ENGINEER</div>', unsafe_allow_html=True)
    st.subheader("Sunny Thakur")
    st.caption("AI & FinTech Risk Infrastructure Developer")
    st.divider()
    
    st.markdown("### 🖥️ Engine Specifications")
    st.write("**Core Framework:** FastAPI v2.0")
    st.write("**Runtime:** Python 3.14 / Torch 2.x")
    st.write("**Persistence:** SQLite / SQLAlchemy ORM")
    st.write("**Inference Mode:** Synchronous & Batch Vectorized")
    st.divider()
    
    st.markdown("### 🧠 Model Zoo Telemetry")
    with st.expander("1. Supervised Classifier (XGBoost)", expanded=True):
        st.markdown("""
        - **Objective:** `binary:logistic`
        - **Input Features:** 4 Tabular Features
        - **Inference Latency:** ~2.1ms
        - **Threshold Tier:**
          - Prob > 0.65: Severe Risk
          - Prob > 0.35: Moderate Risk
        """)
        
    with st.expander("2. Behavioral Autoencoder (PyTorch)", expanded=False):
        st.markdown("""
        - **Architecture:** 4 -> 8 -> 2 -> 8 -> 4 (Linear + ReLU)
        - **Loss Objective:** Reconstruction MSE
        - **Latent Bottleneck:** 2 Dimensions
        - **Threshold Tier:**
          - Loss > 1.8: Anomaly Outlier
          - Loss > 0.9: Behavioral Shift
        """)
        
    st.divider()
    st.caption("SentinelPay Core Engine © 2026. All Rights Reserved.")

# --- Top Header & Operational Banner ---
c_title, c_badge = st.columns([3, 1])
with c_title:
    st.title("🛡️ SentinelPay Risk Intelligence Gateway")
    st.markdown("Enterprise Hybrid Fraud Scoring, Behavioral Reconstruction & Regulatory RAG Reporting")
with c_badge:
    st.write("")
    st.success("🟢 Core Gateway: ACTIVE (Port 8000)")

st.divider()

# --- Main Navigation Tabs ---
tab_live, tab_batch, tab_ledger, tab_architecture = st.tabs([
    "⚡ Live Risk Simulation & SAR", 
    "📁 High-Volume Batch Ingestion", 
    "📋 Compliance Audit Ledger",
    "🔍 System Architecture & Model Blueprint"
])

# ==========================================
# TAB 1: Live Risk Simulation & SAR
# ==========================================
with tab_live:
    col_input, col_report = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Transaction Simulation Gateway")
        st.caption("Provide incoming point-of-sale payload to trigger dual-tier inference.")
        
        acc = st.text_input("Account Identifier / Card Token", "ACC-78291")
        amt = st.number_input("Transaction Volume ($)", min_value=1.0, max_value=25000.0, value=750.0, step=10.0)
        
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            hr = st.slider("Execution Hour (0-23)", 0, 23, 2)
        with col_sub2:
            vel = st.number_input("Velocity (24h Tx Count)", min_value=0, max_value=100, value=8)
            
        dist = st.number_input("Distance from Baseline Cluster (km)", min_value=0.0, max_value=5000.0, value=140.0, step=5.0)
        
        submit_btn = st.button("Execute Risk & Compliance Protocol", type="primary", use_container_width=True)

    with col_report:
        st.subheader("Forensic Investigation Dossier (RAG Synthesis)")
        if submit_btn:
            payload = {
                "account_id": acc,
                "amount": amt,
                "hour_of_day": hr,
                "distance_from_home_km": dist,
                "velocity_last_24h": vel
            }
            try:
                res = requests.post("http://127.0.0.1:8000/api/v1/evaluate-transaction", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    dec = data["decision"]
                    
                    if dec == "APPROVED":
                        st.success(f"**Gateway Verdict: {dec}** — {data['reason']}")
                    elif dec == "FLAGGED_REVIEW":
                        st.warning(f"**Gateway Verdict: {dec}** — {data['reason']}")
                    else:
                        st.error(f"**Gateway Verdict: {dec}** — {data['reason']}")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("ML Fraud Probability (XGBoost)", f"{data['ml_fraud_probability']*100:.1f}%")
                    m2.metric("DL Anomaly Loss (MSE)", f"{data['dl_anomaly_score']}")
                    
                    st.markdown("##### ⚖️ Triggered Regulatory Breaches")
                    if data["regulatory_citations"]:
                        for cite in data["regulatory_citations"]:
                            st.info(cite)
                    else:
                        st.caption("No specific AML threshold triggers detected.")
                        
                    st.markdown("##### 📑 Automated Suspicious Activity Report (SAR)")
                    st.code(data["forensic_sar_report"], language="markdown")
                else:
                    st.error(f"Engine Protocol Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Uvicorn API Gateway is offline. Ensure 'uvicorn main:app --reload' is running.")
        else:
            st.info("Awaiting execution parameters from terminal interface.")

# ==========================================
# TAB 2: Batch CSV Ingestion
# ==========================================
with tab_batch:
    st.subheader("Bulk Ingestion & Stream Vectorization")
    st.caption("Ingest large-scale batch transaction streams for automated risk classification and ledger persistence.")
    
    uploaded_file = st.file_uploader("Upload CSV Transaction Feed", type=["csv"])
    if uploaded_file is not None:
        st.success(f"Source file '{uploaded_file.name}' staged for processing.")
        if st.button("Execute Pipeline Vector Processing", type="primary"):
            with st.spinner("Streaming records through inference matrix..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    res = requests.post("http://127.0.0.1:8000/api/v1/batch-evaluate-csv", files=files)
                    
                    if res.status_code == 200:
                        batch_res = res.json()
                        summary = batch_res["summary"]
                        
                        st.success("Batch operations successfully finalized.")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Total Processed", summary["total_processed"])
                        s2.metric("Approved", summary["approved_count"])
                        s3.metric("Flagged Review", summary["flagged_count"])
                        s4.metric("Declined", summary["declined_count"])
                        
                        st.markdown("##### Ingested Telemetry Feed (Preview)")
                        st.dataframe(pd.DataFrame(batch_res["sample_records"]), use_container_width=True)
                    else:
                        st.error(f"Batch Processing Failed: {res.text}")
                except Exception as e:
                    st.error(f"Ingestion gateway failure: {e}")

# ==========================================
# TAB 3: Audit Ledger Table
# ==========================================
with tab_ledger:
    st.subheader("Regulatory Audit Ledger & Records")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        search_account = st.text_input("Filter Account ID", "")
    with c2:
        filter_status = st.selectbox("Filter Disposition", ["ALL", "APPROVED", "FLAGGED_REVIEW", "DECLINED"])
    with c3:
        st.write("")
        st.write("")
        refresh = st.button("Refresh Feed", use_container_width=True)
        
    try:
        url = "http://127.0.0.1:8000/api/v1/ledger?"
        if search_account.strip():
            url += f"account_id={search_account.strip()}&"
        if filter_status != "ALL":
            url += f"status={filter_status}&"
            
        res = requests.get(url)
        if res.status_code == 200:
            records = res.json()
            if records:
                df = pd.DataFrame(records)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="📥 Export Ledger Snapshot (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="sentinelpay_audit_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No audit logs available for selected filters.")
    except Exception as e:
        st.error(f"Ledger connectivity issue: {e}")

# ==========================================
# TAB 4: Model Architecture & Blueprint
# ==========================================
with tab_architecture:
    st.subheader("System Architecture & AI Defense Blueprint")
    st.caption("Dual-Tier Hybrid Decision Engine designed by Sunny Thakur.")
    
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("### 🏛️ Engineering Stack")
        st.markdown("""
        * **Author / Developer:** Sunny Thakur
        * **API Gateway:** FastAPI with asynchronous ASGI lifecycle handlers
        * **Supervised Machine Learning:** XGBoost (Gradient Boosted Decision Trees)
        * **Unsupervised Deep Learning:** PyTorch Deep Autoencoder (Encoder-Decoder Architecture)
        * **Compliance Synthesis:** Retrieval-Augmented Generation (RAG) referencing AML Policy Standards
        * **Database & Persistence:** SQLite via SQLAlchemy Declarative ORM
        """)
        
    with arch_col2:
        st.markdown("### ⚙️ Decision Protocol Matrix")
        st.markdown("""
        | Condition | Risk Category | Action |
        | :--- | :--- | :--- |
        | `ML Prob > 0.65` OR `DL Loss > 1.8` | Severe / Critical | **DECLINED** (Mandatory SAR Triggered) |
        | `ML Prob > 0.35` OR `DL Loss > 0.9` | Moderate / Anomalous | **FLAGGED_REVIEW** (2FA Challenge) |
        | Default Tolerances | Low / Expected Baseline | **APPROVED** (Frictionless Clearing) |
        """)
        
    st.info("💡 **Why Hybrid AI?** Supervised models (XGBoost) excel at catching known fraud patterns seen during training, while Deep Autoencoders measure reconstruction error to identify novel, zero-day fraud tactics without prior labels.")