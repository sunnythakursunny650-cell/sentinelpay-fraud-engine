# main.py

from contextlib import asynccontextmanager
import io
import json

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


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# PYTORCH AUTOENCODER
# ============================================================

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


# ============================================================
# MODEL STORAGE
# ============================================================

models = {}


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n" + "=" * 60)
    print(" SENTINELPAY - LOADING MODEL ENGINE")
    print("=" * 60)

    # Load StandardScaler
    models["scaler"] = joblib.load(
        "saved_models/scaler.joblib"
    )

    print("✓ StandardScaler loaded")

    # Load XGBoost
    models["ml_model"] = joblib.load(
        "saved_models/xgb_fraud_model.joblib"
    )

    print("✓ XGBoost model loaded")

    # Load model metrics
    try:

        with open(
            "saved_models/model_metrics.json",
            "r",
            encoding="utf-8"
        ) as f:

            metrics = json.load(f)

        # IMPORTANT:
        # These keys match the actual model_metrics.json

        models["ml_threshold"] = float(
            metrics["xgboost"]["classification_threshold"]
        )

        models["dl_threshold"] = float(
            metrics["autoencoder"]["anomaly_threshold"]
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError
    ):

        print(
            "⚠ model_metrics.json not found or incomplete."
        )

        print(
            "⚠ Using fallback thresholds."
        )

        models["ml_threshold"] = 0.70

        models["dl_threshold"] = 1.333665

    # Load PyTorch Autoencoder
    dl_model = TransactionAutoencoder(
        input_dim=4
    )

    dl_model.load_state_dict(
        torch.load(
            "saved_models/autoencoder_model.pth",
            weights_only=True
        )
    )

    dl_model.eval()

    models["dl_model"] = dl_model

    print("✓ PyTorch Autoencoder loaded")

    print(
        f"✓ XGBoost threshold: "
        f"{models['ml_threshold']:.4f}"
    )

    print(
        f"✓ Autoencoder threshold: "
        f"{models['dl_threshold']:.4f}"
    )

    print("=" * 60)
    print(" MODEL ENGINE READY")
    print("=" * 60 + "\n")

    yield

    models.clear()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SentinelPay Risk Engine API",
    description=(
        "Hybrid ML and Deep Learning transaction "
        "risk evaluation API."
    ),
    version="2.1.0",
    lifespan=lifespan
)


# ============================================================
# REQUEST MODEL
# ============================================================

class TransactionPayload(BaseModel):

    account_id: str = Field(
        ...,
        example="ACC-TEST-001"
    )

    amount: float = Field(
        ...,
        gt=0,
        example=450.00
    )

    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        example=3
    )

    distance_from_home_km: float = Field(
        ...,
        ge=0,
        example=85.5
    )

    velocity_last_24h: int = Field(
        ...,
        ge=0,
        example=6
    )


# ============================================================
# DECISION ENGINE
# ============================================================

def determine_decision(
    ml_prob: float,
    dl_loss: float
):

    ml_threshold = models["ml_threshold"]

    dl_threshold = models["dl_threshold"]

    # High risk
    if (
        ml_prob >= ml_threshold
        or dl_loss >= dl_threshold
    ):

        decision = "DECLINED"

        reason = (
            "Critical transaction risk detected "
            "by the hybrid ML/DL risk engine."
        )

    # Moderate risk
    elif (
        ml_prob >= ml_threshold * 0.50
        or dl_loss >= dl_threshold * 0.65
    ):

        decision = "FLAGGED_REVIEW"

        reason = (
            "Transaction requires additional "
            "manual risk review."
        )

    # Low risk
    else:

        decision = "APPROVED"

        reason = (
            "Transaction remains within the "
            "configured risk thresholds."
        )

    return decision, reason


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(
    amount: float,
    hour_of_day: int,
    distance_from_home_km: float,
    velocity_last_24h: int
):

    raw_features = np.array([
        [
            amount,
            hour_of_day,
            distance_from_home_km,
            velocity_last_24h
        ]
    ])

    # Scale features
    scaled_features = models[
        "scaler"
    ].transform(raw_features)

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    ml_prob = float(
        models["ml_model"]
        .predict_proba(
            scaled_features
        )[0][1]
    )

    # --------------------------------------------------------
    # PYTORCH AUTOENCODER
    # --------------------------------------------------------

    tensor_features = torch.tensor(
        scaled_features,
        dtype=torch.float32
    )

    with torch.no_grad():

        reconstructed = models[
            "dl_model"
        ](tensor_features)

        dl_loss = float(
            nn.MSELoss()(
                reconstructed,
                tensor_features
            ).item()
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision, reason = determine_decision(
        ml_prob,
        dl_loss
    )

    return (
        ml_prob,
        dl_loss,
        decision,
        reason
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return {

        "status": "Online",

        "service":
            "SentinelPay Risk Engine",

        "version":
            "2.1.0",

        "model_engine":
            "XGBoost + PyTorch Autoencoder",

        "ml_threshold":
            round(
                models.get(
                    "ml_threshold",
                    0.70
                ),
                4
            ),

        "dl_threshold":
            round(
                models.get(
                    "dl_threshold",
                    1.333665
                ),
                4
            )
    }


# ============================================================
# SINGLE TRANSACTION EVALUATION
# ============================================================

@app.post(
    "/api/v1/evaluate-transaction"
)
def evaluate_transaction(
    data: TransactionPayload,
    db: Session = Depends(get_db)
):

    # Model prediction
    (
        ml_prob,
        dl_loss,
        decision,
        reason
    ) = predict_transaction(

        amount=data.amount,

        hour_of_day=data.hour_of_day,

        distance_from_home_km=
            data.distance_from_home_km,

        velocity_last_24h=
            data.velocity_last_24h
    )

    # Policy / compliance review
    rag_result = generate_forensic_report(

        account_id=data.account_id,

        amount=data.amount,

        hour=data.hour_of_day,

        distance_km=
            data.distance_from_home_km,

        velocity=
            data.velocity_last_24h,

        ml_prob=ml_prob,

        dl_score=dl_loss,

        decision=decision
    )

    # Save transaction
    record = TransactionAudit(

        account_id=data.account_id,

        amount=data.amount,

        hour=data.hour_of_day,

        distance_km=
            data.distance_from_home_km,

        velocity_last_24h=
            data.velocity_last_24h,

        ml_fraud_probability=
            round(
                ml_prob,
                4
            ),

        dl_anomaly_loss=
            round(
                dl_loss,
                4
            ),

        decision=decision,

        reason=reason
    )

    db.add(record)

    db.commit()

    db.refresh(record)

    # API response
    return {

        "transaction_id":
            record.id,

        "account_id":
            record.account_id,

        "decision":
            decision,

        "reason":
            reason,

        "ml_fraud_probability":
            round(
                ml_prob,
                4
            ),

        "dl_anomaly_score":
            round(
                dl_loss,
                4
            ),

        "thresholds": {

            "ml_fraud_threshold":
                round(
                    models["ml_threshold"],
                    4
                ),

            "dl_anomaly_threshold":
                round(
                    models["dl_threshold"],
                    4
                )
        },

        "policy_citations":
            rag_result["citations"],

        "compliance_review_report":
            rag_result["audit_report"]
    }


# ============================================================
# BATCH CSV EVALUATION
# ============================================================

@app.post(
    "/api/v1/batch-evaluate-csv"
)
async def batch_evaluate_csv(

    file: UploadFile = File(...),

    db: Session = Depends(get_db)
):

    # Validate filename
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="CSV file is required."
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )

    # Read file
    contents = await file.read()

    try:

        df = pd.read_csv(
            io.StringIO(
                contents.decode("utf-8")
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV: {str(e)}"
        )

    # Required columns
    required_columns = [

        "account_id",

        "amount",

        "hour_of_day",

        "distance_from_home_km",

        "velocity_last_24h"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise HTTPException(

            status_code=400,

            detail=(
                "Missing required columns: "
                f"{missing_columns}"
            )
        )

    if len(df) == 0:

        raise HTTPException(

            status_code=400,

            detail="CSV file contains no records."
        )

    # Numeric validation
    numeric_columns = [

        "amount",

        "hour_of_day",

        "distance_from_home_km",

        "velocity_last_24h"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"
        )

    if df[numeric_columns].isnull().any().any():

        raise HTTPException(

            status_code=400,

            detail=(
                "CSV contains invalid or "
                "missing numeric values."
            )
        )

    # Feature matrix
    features = df[

        [

            "amount",

            "hour_of_day",

            "distance_from_home_km",

            "velocity_last_24h"
        ]

    ].values

    scaled = models[
        "scaler"
    ].transform(features)

    # XGBoost predictions
    ml_probs = models[
        "ml_model"
    ].predict_proba(
        scaled
    )[:, 1]

    # Autoencoder predictions
    tensor_batch = torch.tensor(

        scaled,

        dtype=torch.float32
    )

    with torch.no_grad():

        reconstructed = models[
            "dl_model"
        ](tensor_batch)

        dl_losses = torch.mean(

            (
                reconstructed
                - tensor_batch
            ) ** 2,

            dim=1

        ).numpy()

    # Process results
    results = []

    for i, row in df.iterrows():

        ml_prob = float(
            ml_probs[i]
        )

        dl_loss = float(
            dl_losses[i]
        )

        decision, reason = (
            determine_decision(
                ml_prob,
                dl_loss
            )
        )

        # Save database record
        record = TransactionAudit(

            account_id=str(
                row["account_id"]
            ),

            amount=float(
                row["amount"]
            ),

            hour=int(
                row["hour_of_day"]
            ),

            distance_km=float(
                row["distance_from_home_km"]
            ),

            velocity_last_24h=int(
                row["velocity_last_24h"]
            ),

            ml_fraud_probability=round(
                ml_prob,
                4
            ),

            dl_anomaly_loss=round(
                dl_loss,
                4
            ),

            decision=decision,

            reason=reason
        )

        db.add(record)

        results.append({

            "account_id":
                str(row["account_id"]),

            "amount":
                float(row["amount"]),

            "decision":
                decision,

            "ml_prob":
                round(
                    ml_prob,
                    4
                ),

            "dl_loss":
                round(
                    dl_loss,
                    4
                )
        })

    db.commit()

    # Summary
    summary = {

        "total_processed":
            len(results),

        "approved_count":
            sum(
                1
                for result in results
                if result["decision"]
                == "APPROVED"
            ),

        "flagged_count":
            sum(
                1
                for result in results
                if result["decision"]
                == "FLAGGED_REVIEW"
            ),

        "declined_count":
            sum(
                1
                for result in results
                if result["decision"]
                == "DECLINED"
            )
    }

    return {

        "summary":
            summary,

        "thresholds": {

            "ml_fraud_threshold":
                round(
                    models["ml_threshold"],
                    4
                ),

            "dl_anomaly_threshold":
                round(
                    models["dl_threshold"],
                    4
                )
        },

        "sample_records":
            results[:10]
    }


# ============================================================
# AUDIT LEDGER
# ============================================================

@app.get(
    "/api/v1/ledger"
)
def get_ledger(

    account_id: str = Query(
        None
    ),

    status: str = Query(
        None
    ),

    db: Session = Depends(get_db)
):

    query = db.query(
        TransactionAudit
    )

    # Account filter
    if account_id:

        query = query.filter(

            TransactionAudit.account_id.ilike(

                f"%{account_id}%"
            )
        )

    # Status filter
    if status:

        query = query.filter(

            TransactionAudit.decision
            == status.upper()
        )

    records = (

        query

        .order_by(
            TransactionAudit.id.desc()
        )

        .all()
    )

    return [

        {

            "id":
                record.id,

            "account_id":
                record.account_id,

            "amount":
                f"${record.amount:.2f}",

            "ml_prob":
                record.ml_fraud_probability,

            "dl_anomaly":
                record.dl_anomaly_loss,

            "decision":
                record.decision,

            "reason":
                record.reason,

            "timestamp":

                (
                    record.created_at.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    if record.created_at

                    else None
                )
        }

        for record in records
    ]