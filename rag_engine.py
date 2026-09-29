# rag_engine.py
import os
import re

POLICY_FILE = "compliance_policy.txt"


def load_policy_sections():
  """Parses compliance_policy.txt into structured sections."""
  if not os.path.exists(POLICY_FILE):
    return []

  with open(POLICY_FILE, "r", encoding="utf-8") as f:
    content = f.read()

  # Split sections based on [SECTION-XXX] tags
  raw_sections = re.split(r"(\[SECTION-\d+:.*?\])", content)
  sections = []
  for i in range(1, len(raw_sections), 2):
    header = raw_sections[i].strip()
    body = raw_sections[i + 1].strip() if i + 1 < len(raw_sections) else ""
    sections.append({"header": header, "body": body, "full_text": f"{header}\n{body}"})
  return sections


def retrieve_relevant_policies(amount: float, distance_km: float, velocity: int):
  """RAG Retriever: Matches transaction risk parameters against policy rules."""
  sections = load_policy_sections()
  matched = []

  for sec in sections:
    # Rule matching based on operational thresholds
    if "HIGH-VELOCITY" in sec["header"] and velocity > 5:
      matched.append(sec)
    elif "GEOGRAPHIC LOCATION" in sec["header"] and distance_km > 100:
      matched.append(sec)
    elif "UNUSUAL TICKET SIZE" in sec["header"] and amount > 500:
      matched.append(sec)

  if not matched:
    # If no violation found, return baseline clearance rule
    for sec in sections:
      if "BASELINE" in sec["header"]:
        matched.append(sec)
        break

  return matched


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
  """RAG Generator: Produces a structured Financial Crime Compliance Audit Report."""
  matched_policies = retrieve_relevant_policies(amount, distance_km, velocity)
  policy_citations = [m["header"] for m in matched_policies]
  policy_context = "\n\n".join([m["full_text"] for m in matched_policies])

  # Structured enterprise SAR (Suspicious Activity Report) synthesis
  summary = (
      f"FORENSIC AUDIT DOSSIER: Transaction for Account '{account_id}' evaluated at {decision}.\n"
      f"• Ticket Value: ${amount:.2f} | Execution Time: {hour:02d}:00 HRS\n"
      f"• Anomaly Metrics: ML Fraud Likelihood: {ml_prob*100:.1f}%, Behavioral DL Reconstruction Loss: {dl_score:.2f}\n"
      f"• Risk Velocity Factor: {velocity} tx/24h | Geographical Deviation: {distance_km:.1f} km from registered cluster.\n\n"
      f"REGULATORY BREACH CITATIONS:\n{policy_context}\n\n"
      f"MANDATED COMPLIANCE DIRECTIVE:\n"
  )

  if decision == "DECLINED":
    summary += (
        "Immediate asset-hold protocol enacted. Account token placed on restricted freeze. "
        "Automated SAR filing initiated for regulatory submission under AML FinTech Mandate."
    )
  elif decision == "FLAGGED_REVIEW":
    summary += (
        "Tier-2 Manual Forensic Review queued. Two-factor behavioral verification SMS/Token dispatched "
        "prior to transaction clearing."
    )
  else:
    summary += (
        "Transaction conforms within standard clearing tolerances. Logged in audit ledger with zero-friction release."
    )

  return {
      "citations": policy_citations,
      "audit_report": summary,
  }