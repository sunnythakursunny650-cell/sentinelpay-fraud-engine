import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="SentinelPay | Risk Intelligence Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        '<div class="author-badge">PROJECT DEVELOPER</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Sunny Thakur")
    st.caption("Python | Machine Learning | Risk Analytics")

    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github)]"
        "(https://github.com/sunnythakursunny650-cell)"
    )
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)]"
        "(https://www.linkedin.com/in/sunny-thakur-4a56103b9/)"
    )

    st.divider()

    st.markdown("### 🖥️ Engine Specifications")
    st.write("**API Framework:** FastAPI")
    st.write("**API Version:** 2.1.0")
    st.write("**Runtime:** Python / PyTorch")
    st.write("**Persistence:** SQLite / SQLAlchemy ORM")
    st.write("**Inference Mode:** Single Transaction + Batch CSV")

    st.divider()

    st.markdown("### 🧠 Model Configuration")

    with st.expander("1. Supervised Classifier (XGBoost)", expanded=True):
        st.markdown(
            """
            - **Objective:** Binary fraud classification
            - **Input Features:** 4 tabular transaction features
            - **Classification Threshold:** `0.70`
            - **Evaluation:** Precision, Recall, F1, ROC-AUC
            """
        )

    with st.expander("2. Behavioral Autoencoder (PyTorch)", expanded=False):
        st.markdown(
            """
            - **Architecture:** `4 → 8 → 2 → 8 → 4`
            - **Loss Objective:** Reconstruction MSE
            - **Latent Bottleneck:** 2 dimensions
            - **Anomaly Threshold:** `1.3337`
            """
        )

    st.divider()
    st.caption("SentinelPay is an illustrative ML/DL transaction-risk demonstration.")

# -----------------------------
# Header
# -----------------------------
c_title, c_badge = st.columns([3, 1])

with c_title:
    st.title("🛡️ SentinelPay Risk Intelligence Gateway")
    st.markdown(
        "Hybrid Machine Learning + Deep Learning transaction-risk "
        "evaluation with policy retrieval and audit logging."
    )

with c_badge:
    st.write("")
    st.success("🟢 API Gateway: ACTIVE")

st.divider()

# -----------------------------
# Navigation
# -----------------------------
tab_live, tab_batch, tab_ledger, tab_architecture = st.tabs(
    [
        "⚡ Live Risk Simulation",
        "📁 Batch CSV Processing",
        "📋 Audit Ledger",
        "🔍 System Architecture",
    ]
)

# ==========================================================
# TAB 1: LIVE RISK SIMULATION
# ==========================================================
with tab_live:
    col_input, col_report = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Transaction Simulation")

        st.caption(
            "Enter transaction features and send them to the SentinelPay API."
        )

        acc = st.text_input("Account Identifier", "ACC-78291")

        amt = st.number_input(
            "Transaction Amount ($)",
            min_value=1.0,
            max_value=25000.0,
            value=750.0,
            step=10.0,
        )

        col_sub1, col_sub2 = st.columns(2)

        with col_sub1:
            hr = st.slider("Transaction Hour (0-23)", 0, 23, 2)

        with col_sub2:
            vel = st.number_input(
                "Transaction Velocity (24h)",
                min_value=0,
                max_value=100,
                value=8,
            )

        dist = st.number_input(
            "Distance from Baseline (km)",
            min_value=0.0,
            max_value=5000.0,
            value=140.0,
            step=5.0,
        )

        submit_btn = st.button(
            "Execute Risk Evaluation",
            type="primary",
            use_container_width=True,
        )

    with col_report:
        st.subheader("Risk Evaluation & Policy Review")

        if submit_btn:
            payload = {
                "account_id": acc,
                "amount": amt,
                "hour_of_day": hr,
                "distance_from_home_km": dist,
                "velocity_last_24h": vel,
            }

            try:
                res = requests.post(
                    f"{API_BASE_URL}/api/v1/evaluate-transaction",
                    json=payload,
                    timeout=30,
                )

                if res.status_code == 200:
                    data = res.json()
                    dec = data["decision"]

                    if dec == "APPROVED":
                        st.success(
                            f"**Decision: {dec}** — {data['reason']}"
                        )
                    elif dec == "FLAGGED_REVIEW":
                        st.warning(
                            f"**Decision: {dec}** — {data['reason']}"
                        )
                    else:
                        st.error(
                            f"**Decision: {dec}** — {data['reason']}"
                        )

                    m1, m2 = st.columns(2)

                    m1.metric(
                        "XGBoost Fraud Probability",
                        f"{data['ml_fraud_probability'] * 100:.1f}%",
                    )

                    m2.metric(
                        "Autoencoder Anomaly Score",
                        f"{data['dl_anomaly_score']:.4f}",
                    )

                    st.markdown("##### 📌 Matched Policy Sections")

                    policy_citations = data.get("policy_citations", [])

                    if policy_citations:
                        for cite in policy_citations:
                            st.info(cite)
                    else:
                        st.caption(
                            "No specific policy section matched this transaction."
                        )

                    st.markdown("##### 📑 Automated Compliance Review Report")

                    st.code(
                        data.get(
                            "compliance_review_report",
                            "No compliance review report returned.",
                        ),
                        language="text",
                    )

                    st.caption(
                        "This report is generated from the project's "
                        "illustrative internal policy rules and is not a "
                        "regulatory filing or legal determination."
                    )

                else:
                    st.error(f"API Error {res.status_code}: {res.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI server is offline. Start it with:\n\n"
                    "`uvicorn main:app --reload`"
                )

            except requests.exceptions.Timeout:
                st.error("The API request timed out.")

            except Exception as e:
                st.error(f"Unexpected error: {e}")

        else:
            st.info("Enter transaction details and click Execute Risk Evaluation.")

# ==========================================================
# TAB 2: BATCH CSV PROCESSING
# ==========================================================
with tab_batch:
    st.subheader("Batch Transaction Processing")

    st.caption(
        "Upload a CSV file and process multiple transactions through "
        "the same hybrid ML/DL decision engine."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Transaction File",
        type=["csv"],
    )

    if uploaded_file is not None:
        st.success(
            f"File '{uploaded_file.name}' is ready for processing."
        )

        if st.button(
            "Execute Batch Evaluation",
            type="primary",
        ):
            with st.spinner("Processing transactions..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv",
                        )
                    }

                    res = requests.post(
                        f"{API_BASE_URL}/api/v1/batch-evaluate-csv",
                        files=files,
                        timeout=120,
                    )

                    if res.status_code == 200:
                        batch_res = res.json()
                        summary = batch_res["summary"]

                        st.success("Batch processing completed successfully.")

                        s1, s2, s3, s4 = st.columns(4)

                        s1.metric(
                            "Total Processed",
                            summary["total_processed"],
                        )
                        s2.metric(
                            "Approved",
                            summary["approved_count"],
                        )
                        s3.metric(
                            "Flagged Review",
                            summary["flagged_count"],
                        )
                        s4.metric(
                            "Declined",
                            summary["declined_count"],
                        )

                        st.markdown("##### Processed Records Preview")

                        st.dataframe(
                            pd.DataFrame(batch_res["sample_records"]),
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:
                        st.error(
                            f"Batch Processing Failed "
                            f"{res.status_code}: {res.text}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "FastAPI server is offline. Start it with:\n\n"
                        "`uvicorn main:app --reload`"
                    )

                except requests.exceptions.Timeout:
                    st.error("Batch request timed out.")

                except Exception as e:
                    st.error(f"Batch processing error: {e}")

# ==========================================================
# TAB 3: AUDIT LEDGER
# ==========================================================
with tab_ledger:
    st.subheader("Transaction Audit Ledger")

    st.caption(
        "View transaction-risk decisions stored by the SentinelPay API."
    )

    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        search_account = st.text_input(
            "Filter Account ID",
            "",
        )

    with c2:
        filter_status = st.selectbox(
            "Filter Decision",
            ["ALL", "APPROVED", "FLAGGED_REVIEW", "DECLINED"],
        )

    with c3:
        st.write("")
        st.write("")
        refresh = st.button(
            "Refresh Feed",
            use_container_width=True,
        )

    # Refresh button is intentionally available; data also loads on tab open.
    try:
        params = {}

        if search_account.strip():
            params["account_id"] = search_account.strip()

        if filter_status != "ALL":
            params["status"] = filter_status

        res = requests.get(
            f"{API_BASE_URL}/api/v1/ledger",
            params=params,
            timeout=30,
        )

        if res.status_code == 200:
            records = res.json()

            if records:
                df = pd.DataFrame(records)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    label="📥 Export Ledger Snapshot (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="sentinelpay_audit_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info(
                    "No audit records found for the selected filters."
                )

        else:
            st.error(
                f"Ledger API Error {res.status_code}: {res.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI server is offline. Start it with:\n\n"
            "`uvicorn main:app --reload`"
        )

    except Exception as e:
        st.error(f"Ledger connectivity issue: {e}")

# ==========================================================
# TAB 4: SYSTEM ARCHITECTURE
# ==========================================================
with tab_architecture:
    st.subheader("System Architecture & ML/DL Blueprint")

    st.caption(
        "Hybrid transaction-risk demonstration system developed by Sunny Thakur."
    )

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("### 🏛️ Engineering Stack")

        st.markdown(
            """
            * **Developer:** [Sunny Thakur](https://www.linkedin.com/in/sunny-thakur-4a56103b9/)
            * **API:** FastAPI
            * **Supervised ML:** XGBoost
            * **Deep Learning:** PyTorch Autoencoder
            * **Policy Retrieval:** Rule-based section matching
            * **Persistence:** SQLite + SQLAlchemy
            * **Dashboard:** Streamlit
            * **Data:** Synthetic transaction data for demonstration
            """
        )

    with arch_col2:
        st.markdown("### ⚙️ Decision Protocol")

        st.markdown(
            """
            | Condition | Decision |
            | :--- | :--- |
            | `ML Prob ≥ 0.70` OR `DL Loss ≥ 1.3337` | **DECLINED** |
            | `ML Prob ≥ 0.35` OR `DL Loss ≥ 0.8669` | **FLAGGED_REVIEW** |
            | Otherwise | **APPROVED** |
            """
        )

    st.info(
        "💡 **Why Hybrid ML + DL?** XGBoost provides a supervised fraud-risk "
        "signal from transaction features, while the autoencoder provides an "
        "independent reconstruction-error signal for unusual behavior. "
        "The final decision combines both configured signals."
    )

    st.warning(
        "Demo limitation: the models were trained and evaluated on synthetic "
        "transaction data. The policy rules are illustrative internal demo "
        "rules and should not be presented as official regulatory guidance."
    )