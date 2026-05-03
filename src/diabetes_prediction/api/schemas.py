from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class SmokingHistory(str, Enum):
    never = "never"
    ever = "ever"
    current = "current"
    former = "former"
    not_current = "not current"
    no_info = "No Info"

class PatientData(BaseModel):
    gender: str
    age: float = Field(ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: SmokingHistory
    bmi: float = Field(gt=0)
    HbA1c_level: float = Field(gt=0)
    blood_glucose_level: int = Field(gt=0)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

class ExplainRequest(BaseModel):
    data: PatientData
    prediction: int
    probability: float    