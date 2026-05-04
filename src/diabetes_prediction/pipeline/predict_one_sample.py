from pathlib import Path
import sys
import joblib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from diabetes_prediction.transformation.transformation import DataTransformation


def predict_single_sample(user_input: dict) -> dict:
    """
    Predict diabetes for a single raw sample using the notebook-trained
    preprocessor (preprocessor.pkl) and Random Forest model (RF_model.pkl).

    This bypasses the full pipeline and directly uses the same artifacts
    that the Random_Forest.ipynb notebook produced, ensuring identical
    preprocessing and prediction behaviour.
    """
    try:
       
        preprocessor_path = PROJECT_ROOT / "notebooks" / "Transformation" / "preprocessor.pkl"
        model_path = PROJECT_ROOT / "models" / "RF_model.pkl"

        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)

        
        transformer = DataTransformation()
        transformer.preprocessor = preprocessor          # use the saved preprocessor

        X_sample = transformer.transform_one(user_input)  # prepare_features + transform

       
        cols_to_drop = [
            "age",
            "blood_glucose_level",
            "age_bmi_interaction",
            "age_hba1c_interaction",
            "hypertension",
        ]
        
        cols_to_drop = [c for c in cols_to_drop if c in X_sample.columns]
        X_sample = X_sample.drop(columns=cols_to_drop)

        
        prediction = int(model.predict(X_sample)[0])

        result = {"prediction": prediction}

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_sample)[0][1])
            result["diabetes_probability"] = probability

        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}
    

if __name__ == "__main__":
    sample_input = {
        "gender": "Male",
        "age": 45,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 28.5,
        "HbA1c_level": 6.2,
        "blood_glucose_level": 140
    }

    output = predict_single_sample(sample_input)
    print(output)


