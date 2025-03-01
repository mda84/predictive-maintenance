import os
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./maintenance.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(String, index=True)
    sensor_data = Column(Text)
    prediction = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")
