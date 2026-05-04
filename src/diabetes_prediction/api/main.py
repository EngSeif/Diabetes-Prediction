import os

from fastapi import FastAPI
from diabetes_prediction.api.schemas import PatientData, PredictionResponse
from diabetes_prediction.pipeline.predict_one_sample import predict_single_sample

from fastapi.middleware.cors import CORSMiddleware
import logging


#from diabetes_prediction.api.llm import explain_prediction




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
    try:
        # Convert Pydantic model to dict
        user_input = data.dict()
        
        # Call prediction function
        result = predict_single_sample(user_input)

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Received: {data}")

        if result["success"]:
            prediction = result["result"]["prediction"]
            probability = result["result"].get("diabetes_probability", 0.0)
            return {"prediction": prediction, "probability": probability}
        else:
            # If something went wrong 
            return {"prediction": -1, "probability": 0.0}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"prediction": -1, "probability": 0.0}

   