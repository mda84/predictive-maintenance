import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Header
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np
from transformers import pipeline  # For a fallback LLM or similar usage
from database import engine, SessionLocal, Conversation, Base
from sqlalchemy.orm import Session
from datetime import datetime

# Create database tables.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Predictive Maintenance Chatbot API")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API key authentication.
API_KEY = os.getenv("API_KEY", "secret-key")
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

# Dummy function for model inference.
def predict_failure(input_features: np.ndarray) -> float:
    # Load model, process input_features, and return a failure probability.
    # For simplicity, we return a random value.
    return float(np.random.rand())

class PredictionRequest(BaseModel):
    sensor_readings: list  # List of sensor values
    equipment_id: str

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_endpoint(request: PredictionRequest, db: Session = Depends(get_db)):
    input_features = np.array(request.sensor_readings, dtype=np.float32)
    prediction = predict_failure(input_features)
    # Log prediction.
    conv = Conversation(
        session_id=request.equipment_id,
        user_message=str(request.sensor_readings),
        bot_response=str(prediction),
        timestamp=datetime.utcnow()
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"equipment_id": request.equipment_id, "failure_probability": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
