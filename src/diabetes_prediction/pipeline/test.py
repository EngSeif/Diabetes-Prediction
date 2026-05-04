from diabetes_prediction.pipeline.pipeline  import predict_single_sample

sample_input = {
    "gender": "Female",
    "age": 55,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 29.5,
    "HbA1c_level": 6.1,
    "blood_glucose_level": 160
}

print(predict_single_sample(sample_input))
