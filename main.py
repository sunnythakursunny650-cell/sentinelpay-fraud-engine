# main.py
from contextlib import asynccontextmanager
import io
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

from database import engine, Base, get_db, TransactionAudit
from rag_engine import generate_forensic_report

# Initialize database tables
Base.metadata.create_all(bind=engine)

# DL Autoencoder model architecture
class TransactionAutoencoder(nn.Module):
    def __init__(self, input_dim=4):
        super(TransactionAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["scaler"] = joblib.load("saved_models/scaler.joblib")
    models["ml_model"] = joblib.load("saved_models/xgb_fraud_model.joblib")
    
    dl = TransactionAutoencoder(input_dim=4)
    dl.load_state_dict(torch.load("saved_models/autoencoder_model.pth", weights_only=True))
    dl.eval()
    models["dl_model"] = dl
    yield
    models.clear()

app = FastAPI(
    title="SentinelPay Risk Engine API",
    description="Enterprise Real-Time Hybrid ML, DL & GenAI/RAG Financial Compliance Platform",
    lifespan=lifespan
)

class TransactionPayload(BaseModel):
    account_id: str = Field(..., example="ACC-99214")
    amount: float = Field(..., gt=0, example=450.00)
    hour_of_day: int = Field(..., ge=0, le=23, example=3)
    distance_from_home_km: float = Field(..., ge=0, example=85.5)
    velocity_last_24h: int = Field(..., ge=0, example=6)

@app.get("/")
def health_check():
    return {"status": "Online", "service": "SentinelPay Enterprise Gateway", "version": "2.0.0"}

# --- Endpoint 1: Single Transaction with RAG Forensic Audit ---
@app.post("/api/v1/evaluate-transaction")
def evaluate_transaction(data: TransactionPayload, db: Session = Depends(get_db)):
    raw_features = np.array([[data.amount, data.hour_of_day, data.distance_from_home_km, data.velocity_last_24h]])
    
    # 1. Feature scaling
    scaled_features = models["scaler"].transform(raw_features)
    
    # 2. Supervised ML (XGBoost)
    ml_prob = float(models["ml_model"].predict_proba(scaled_features)[0][1])
    
    # 3. Unsupervised DL (PyTorch Autoencoder Loss)
    tensor_feat = torch.tensor(scaled_features, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = models["dl_model"](tensor_feat)
        dl_loss = float(nn.MSELoss()(reconstructed, tensor_feat).item())
        
    # 4. Multi-tier decision rules
    if ml_prob > 0.65 or dl_loss > 1.8:
        decision = "DECLINED"
        reason = "Critical Fraud Signature Detected (High Confidence)"
    elif ml_prob > 0.35 or dl_loss > 0.9:
        decision = "FLAGGED_REVIEW"
        reason = "Anomalous Spending Pattern / Moderate Risk"
    else:
        decision = "APPROVED"
        reason = "Legitimate Transaction Pattern"
        
    # 5. RAG Engine: Retrieve Policy & Generate Forensic SAR Report
    rag_result = generate_forensic_report(
        account_id=data.account_id,
        amount=data.amount,
        hour=data.hour_of_day,
        distance_km=data.distance_from_home_km,
        velocity=data.velocity_last_24h,
        ml_prob=ml_prob,
        dl_score=dl_loss,
        decision=decision
    )

    # 6. Persist audit to Ledger DB
    record = TransactionAudit(
        account_id=data.account_id,
        amount=data.amount,
        hour=data.hour_of_day,
        distance_km=data.distance_from_home_km,
        velocity_last_24h=data.velocity_last_24h,
        ml_fraud_probability=round(ml_prob, 4),
        dl_anomaly_loss=round(dl_loss, 4),
        decision=decision,
        reason=reason
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    return {
        "transaction_id": record.id,
        "account_id": record.account_id,
        "decision": decision,
        "reason": reason,
        "ml_fraud_probability": round(ml_prob, 4),
        "dl_anomaly_score": round(dl_loss, 4),
        "regulatory_citations": rag_result["citations"],
        "forensic_sar_report": rag_result["audit_report"]
    }

# --- Endpoint 2: Batch CSV Processing ---
@app.post("/api/v1/batch-evaluate-csv")
async def batch_evaluate_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    
    required_cols = ["account_id", "amount", "hour_of_day", "distance_from_home_km", "velocity_last_24h"]
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_cols}")
    
    features = df[["amount", "hour_of_day", "distance_from_home_km", "velocity_last_24h"]].values
    scaled = models["scaler"].transform(features)
    
    # ML Batch Predictions
    ml_probs = models["ml_model"].predict_proba(scaled)[:, 1]
    
    # DL Batch Predictions
    tensor_batch = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        recon_batch = models["dl_model"](tensor_batch)
        dl_losses = torch.mean((recon_batch - tensor_batch) ** 2, dim=1).numpy()
        
    results = []
    for i, row in df.iterrows():
        p = float(ml_probs[i])
        loss = float(dl_losses[i])
        
        if p > 0.65 or loss > 1.8:
            dec = "DECLINED"
            res = "Critical Fraud Signature"
        elif p > 0.35 or loss > 0.9:
            dec = "FLAGGED_REVIEW"
            res = "Anomalous Behavioral Pattern"
        else:
            dec = "APPROVED"
            res = "Normal Pattern"
            
        record = TransactionAudit(
            account_id=str(row["account_id"]),
            amount=float(row["amount"]),
            hour=int(row["hour_of_day"]),
            distance_km=float(row["distance_from_home_km"]),
            velocity_last_24h=int(row["velocity_last_24h"]),
            ml_fraud_probability=round(p, 4),
            dl_anomaly_loss=round(loss, 4),
            decision=dec,
            reason=res
        )
        db.add(record)
        results.append({"account_id": row["account_id"], "amount": row["amount"], "decision": dec, "ml_prob": round(p, 4), "dl_loss": round(loss, 4)})
        
    db.commit()
    
    summary = {
        "total_processed": len(results),
        "approved_count": sum(1 for r in results if r["decision"] == "APPROVED"),
        "flagged_count": sum(1 for r in results if r["decision"] == "FLAGGED_REVIEW"),
        "declined_count": sum(1 for r in results if r["decision"] == "DECLINED")
    }
    return {"summary": summary, "sample_records": results[:10]}

# --- Endpoint 3: Ledger History ---
@app.get("/api/v1/ledger")
def get_ledger(account_id: str = Query(None), status: str = Query(None), db: Session = Depends(get_db)):
    query = db.query(TransactionAudit)
    if account_id:
        query = query.filter(TransactionAudit.account_id.ilike(f"%{account_id}%"))
    if status:
        query = query.filter(TransactionAudit.decision == status.upper())
        
    records = query.order_by(TransactionAudit.id.desc()).all()
    return [{
        "id": r.id,
        "account_id": r.account_id,
        "amount": f"${r.amount:.2f}",
        "ml_prob": r.ml_fraud_probability,
        "dl_anomaly": r.dl_anomaly_loss,
        "decision": r.decision,
        "reason": r.reason,
        "timestamp": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None
    } for r in records]