# dashboard.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="SentinelPay - Enterprise Fraud & AI Intelligence",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SentinelPay | Enterprise Fraud, Anomaly & RAG Intelligence")
st.caption("Hybrid ML (XGBoost) + DL (PyTorch Autoencoders) + AML Compliance RAG Engine + SQLite Ledger")
st.divider()

tab_live, tab_batch, tab_ledger = st.tabs([
    "⚡ Real-Time Evaluation & AI Forensic SAR", 
    "📁 Batch CSV Processing Engine", 
    "📋 Compliance Audit Ledger"
])

# ==========================================
# TAB 1: Real-Time Evaluation & RAG Report
# ==========================================
with tab_live:
    col_input, col_report = st.columns([1, 1])
    
    with col_input:
        st.subheader("Simulate Payment Parameters")
        acc = st.text_input("Account Token / ID", "ACC-78291")
        amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=750.0)
        hr = st.slider("Hour of Transaction (0-23)", 0, 23, 2)
        dist = st.number_input("Distance from Regular Geo-Cluster (km)", min_value=0.0, max_value=1000.0, value=140.0)
        vel = st.number_input("Velocity (Count in rolling 24h)", min_value=0, max_value=50, value=8)
        
        submit_btn = st.button("Run Risk & Compliance Evaluation", type="primary", use_container_width=True)

    with col_report:
        st.subheader("AI Forensic Audit Dossier (RAG Synthesis)")
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
                    m2.metric("DL Anomaly Score (MSE Loss)", f"{data['dl_anomaly_score']}")
                    
                    st.markdown("#### 📜 Regulatory Rule Citations")
                    if data["regulatory_citations"]:
                        for cite in data["regulatory_citations"]:
                            st.info(f"⚖️ {cite}")
                    else:
                        st.write("No regulatory thresholds breached.")
                        
                    st.markdown("#### 📑 Automated Suspicious Activity Report (SAR)")
                    st.code(data["forensic_sar_report"], language="markdown")
                else:
                    st.error(f"Server Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("FastAPI Gateway Offline. Terminal mein uvicorn check karein.")
        else:
            st.info("Left panel par transaction parameter submit karein to live RAG forensic analysis trigger hoga.")

# ==========================================
# TAB 2: Batch CSV Ingestion
# ==========================================
with tab_batch:
    st.subheader("Bulk Transaction Stream Upload")
    st.write("Upload high-volume CSV files to evaluate multi-tenant transactions against ML/DL models.")
    
    uploaded_file = st.file_uploader("Select Financial CSV File", type=["csv"])
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' ready for batch ingestion.")
        if st.button("Execute Batch Fraud Scoring", type="primary"):
            with st.spinner("Processing batch records through AI Pipeline..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    res = requests.post("http://127.0.0.1:8000/api/v1/batch-evaluate-csv", files=files)
                    
                    if res.status_code == 200:
                        batch_res = res.json()
                        summary = batch_res["summary"]
                        
                        st.success("Batch evaluation completed and persisted to Ledger!")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Total Ingested", summary["total_processed"])
                        s2.metric("Approved", summary["approved_count"])
                        s3.metric("Flagged Review", summary["flagged_count"])
                        s4.metric("Declined", summary["declined_count"])
                        
                        st.markdown("#### Sample Ingested Evaluations")
                        st.dataframe(pd.DataFrame(batch_res["sample_records"]), use_container_width=True)
                    else:
                        st.error(f"Batch Processing Failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ==========================================
# TAB 3: Audit Ledger Table
# ==========================================
with tab_ledger:
    st.subheader("Audited Financial Ledger")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        search_account = st.text_input("Filter by Account ID", "")
    with c2:
        filter_status = st.selectbox("Filter Status", ["ALL", "APPROVED", "FLAGGED_REVIEW", "DECLINED"])
    with c3:
        st.write(" ")
        st.write(" ")
        refresh = st.button("Refresh Ledger", use_container_width=True)
        
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
                    label="📥 Export Ledger (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="sentinelpay_audit_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No audit records found matching query.")
    except Exception as e:
        st.error(f"Ledger connectivity issue: {e}")