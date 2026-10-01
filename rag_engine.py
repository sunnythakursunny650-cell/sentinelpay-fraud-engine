# rag_engine.py

import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

POLICY_FILE = "compliance_policy.txt"


# ============================================================
# LOAD POLICY SECTIONS
# ============================================================

def load_policy_sections():
    """
    Load and parse structured policy sections
    from compliance_policy.txt.

    Expected format:

    [SECTION-101: HIGH-VELOCITY ANOMALY PROTOCOL]
    Policy text...

    [SECTION-204: GEOGRAPHIC LOCATION DEVIATION POLICY]
    Policy text...
    """

    if not os.path.exists(POLICY_FILE):
        return []


    try:

        with open(
            POLICY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

    except OSError:

        return []


    # --------------------------------------------------------
    # Find section headers
    # --------------------------------------------------------

    pattern = r"(\[SECTION-\d+:[^\]]+\])"

    matches = list(
        re.finditer(
            pattern,
            content
        )
    )


    sections = []


    for index, match in enumerate(matches):

        header = match.group(1).strip()

        start = match.end()


        if index + 1 < len(matches):

            end = matches[
                index + 1
            ].start()

        else:

            end = len(content)


        body = content[
            start:end
        ].strip()


        sections.append({

            "header": header,

            "body": body,

            "full_text":
                f"{header}\n{body}"

        })


    return sections


# ============================================================
# POLICY RETRIEVAL
# ============================================================

def retrieve_relevant_policies(
    amount: float,
    distance_km: float,
    velocity: int
):
    """
    Retrieve policy sections based on
    transaction risk indicators.

    This is rule-based policy retrieval,
    not an LLM-based RAG system.
    """

    sections = load_policy_sections()

    matched = []


    for section in sections:

        header = section[
            "header"
        ].upper()


        # ----------------------------------------------------
        # High velocity
        # ----------------------------------------------------

        if (
            "HIGH-VELOCITY" in header
            and velocity > 5
        ):

            matched.append(
                section
            )


        # ----------------------------------------------------
        # Geographic deviation
        # ----------------------------------------------------

        elif (
            "GEOGRAPHIC LOCATION" in header
            and distance_km > 100
        ):

            matched.append(
                section
            )


        # ----------------------------------------------------
        # Unusual transaction amount
        # ----------------------------------------------------

        elif (
            "UNUSUAL TICKET SIZE" in header
            and amount > 500
        ):

            matched.append(
                section
            )


    # --------------------------------------------------------
    # Baseline policy
    # --------------------------------------------------------

    if not matched:

        for section in sections:

            if "BASELINE" in section[
                "header"
            ].upper():

                matched.append(
                    section
                )

                break


    return matched


# ============================================================
# GENERATE COMPLIANCE REPORT
# ============================================================

def generate_forensic_report(
    account_id: str,
    amount: float,
    hour: int,
    distance_km: float,
    velocity: int,
    ml_prob: float,
    dl_score: float,
    decision: str,
) -> dict:
    """
    Generate a structured compliance review report.

    Important:
    This function generates a report only.
    It does NOT:
      - freeze accounts
      - hold assets
      - send SMS
      - submit SARs
      - contact regulators
    """


    # --------------------------------------------------------
    # Retrieve relevant policies
    # --------------------------------------------------------

    matched_policies = (
        retrieve_relevant_policies(
            amount=amount,
            distance_km=distance_km,
            velocity=velocity
        )
    )


    policy_citations = [

        policy["header"]

        for policy in matched_policies

    ]


    # --------------------------------------------------------
    # Policy context
    # --------------------------------------------------------

    if matched_policies:

        policy_context = "\n\n".join(

            policy["full_text"]

            for policy in matched_policies

        )

    else:

        policy_context = (
            "No specific policy section "
            "was matched."
        )


    # --------------------------------------------------------
    # Risk classification text
    # --------------------------------------------------------

    if decision == "DECLINED":

        action_text = (

            "Recommended Action: "
            "Place the transaction under "
            "manual compliance review and "
            "follow the organization's "
            "applicable escalation procedures."

        )


    elif decision == "FLAGGED_REVIEW":

        action_text = (

            "Recommended Action: "
            "Queue the transaction for "
            "additional manual review and "
            "customer verification according "
            "to applicable procedures."

        )


    else:

        action_text = (

            "Recommended Action: "
            "No additional risk escalation "
            "is indicated by the configured "
            "transaction risk rules. "
            "Retain the transaction in the "
            "audit ledger."

        )


    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    summary = (

        "FORENSIC COMPLIANCE REVIEW REPORT\n"

        "=================================\n\n"

        f"Account ID: {account_id}\n"

        f"Decision: {decision}\n"

        f"Transaction Amount: "
        f"${amount:.2f}\n"

        f"Execution Time: "
        f"{hour:02d}:00 HRS\n"

        f"Transaction Velocity: "
        f"{velocity} tx/24h\n"

        f"Geographical Deviation: "
        f"{distance_km:.1f} km\n\n"


        "MODEL RISK SIGNALS\n"

        "------------------\n"

        f"XGBoost Fraud Probability: "
        f"{ml_prob * 100:.1f}%\n"

        f"Autoencoder Anomaly Score: "
        f"{dl_score:.4f}\n\n"


        "MATCHED POLICY SECTIONS\n"

        "-----------------------\n"

        f"{policy_context}\n\n"


        "COMPLIANCE REVIEW\n"

        "-----------------\n"

        f"{action_text}\n\n"


        "SYSTEM NOTE\n"

        "-----------\n"

        "This report is generated automatically "
        "from the configured transaction-risk "
        "rules and policy sections. It is "
        "provided for review and does not "
        "constitute an actual regulatory filing "
        "or legal determination."

    )


    return {

        "citations":
            policy_citations,

        "audit_report":
            summary

    }