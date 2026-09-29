# database.py
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./sentinelpay_ledger.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TransactionAudit(Base):
  __tablename__ = "transaction_audits"

  id = Column(Integer, primary_key=True, index=True)
  account_id = Column(String(50), index=True)
  amount = Column(Float, nullable=False)
  hour = Column(Integer)
  distance_km = Column(Float)
  velocity_last_24h = Column(Integer)

  # Model evaluation outputs
  ml_fraud_probability = Column(Float)
  dl_anomaly_loss = Column(Float)
  decision = Column(String(20))  # APPROVED, FLAGGED_REVIEW, DECLINED
  reason = Column(String(100))

  created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()