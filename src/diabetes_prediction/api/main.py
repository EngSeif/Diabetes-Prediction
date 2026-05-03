import os

from fastapi import FastAPI
from schemas import PatientData, PredictionResponse

from fastapi.middleware.cors import CORSMiddleware
import logging

from .schemas import PatientData, PredictionResponse, ExplainRequest
from .llm import explain_prediction




app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    print("Received data:", data)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Received: {data}")

    # TEMP response (NOT just print)
    return {
        "prediction": 0,
        "probability": 0.5
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    explanation = explain_prediction(
        req.data.dict(),
        req.prediction,
        req.probability
    )

    return {
        "explanation": explanation
    }