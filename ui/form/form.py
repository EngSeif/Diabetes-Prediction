import streamlit as st
import requests


def form():
    with st.form("my_form"):
        # Text inputs
        age = st.number_input("Age", min_value=0, max_value=120, value=25)

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            smoking_history = st.selectbox(
                "Smoking History",
                ["No Info", "never", "not current", "current", "ever", "former"],
            )

        hypertension = st.radio("Do you have hypertension?", ["Yes", "No"])
        heart_disease = st.radio("Do you have heart disease?", ["Yes", "No"])

        bmi = st.select_slider(
            "BMI",
            options=[round(x * 0.1, 1) for x in range(100, 1000)],
            value=25.0,
        )

        HbA1c_level = st.select_slider(
            "HbA1c Level",
            options=[round(x * 0.1, 1) for x in range(35, 150)],
            value=5.5,
        )

        blood_glucose_level = st.slider(
            "Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=100
        )

        submitted = st.form_submit_button("Submit")

    if submitted:
        payload = {
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": HbA1c_level,
            "blood_glucose_level": blood_glucose_level,
        }

        response = requests.post(
            f"https://eng-seif-diabetes-prediction.hf.space/predict",
            json=payload,
        )
        result = response.json()

        explain_payload = {
            "data": payload,
            "prediction": result["prediction"],
            "probability": result["probability"],
        }

        explain_response = requests.post(
            f"https://eng-seif-diabetes-prediction.hf.space/explain",
            json=explain_payload,
        )
        explanation = explain_response.json()["explanation"]

        st.success("Form submitted!")
        st.subheader("Here Are Your Results")

        col1, col2 = st.columns(2)

        with col1:
            is_diabetic = result["prediction"] == 1
            bg_color = "#E8654C" if is_diabetic else "#4CAF50"
            icon = "fa-circle-exclamation" if is_diabetic else "fa-circle-check"
            label = "Diabetic" if is_diabetic else "Non-Diabetic"

            st.markdown(
                f"""
                <div class="kpi-box" style="
                    background: linear-gradient(135deg, {bg_color}, {bg_color}cc);
                    color: #ffffff;
                    border-radius: 16px;
                    padding: 24px;
                    text-align: center;
                    box-shadow: 0 4px 15px {bg_color}66;
                ">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">
                        <i class="fa-solid {icon}"></i>
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.85; letter-spacing: 1px; text-transform: uppercase;">
                        Diagnosis
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; margin-top: 6px;">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            probability = result["probability"] * 100
            is_high_risk = probability >= 50
            bg_color = "#E8654C" if is_high_risk else "#4C9BE8"
            risk_label = "High Risk" if is_high_risk else "Low Risk"

            st.markdown(
                f"""
                <div class="kpi-box" style="
                    background: linear-gradient(135deg, {bg_color}, {bg_color}cc);
                    color: #ffffff;
                    border-radius: 16px;
                    padding: 24px;
                    text-align: center;
                    box-shadow: 0 4px 15px {bg_color}66;
                ">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">
                        <i class="fa-solid fa-gauge-high"></i>
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.85; letter-spacing: 1px; text-transform: uppercase;">
                        Probability · {risk_label}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; margin-top: 6px;">
                        {probability:.1f}%
                    </div>
                    <div style="
                        margin-top: 12px;
                        background: rgba(255,255,255,0.25);
                        border-radius: 999px;
                        height: 8px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {probability}%;
                            height: 100%;
                            background: #ffffff;
                            border-radius: 999px;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### AI Explanation")
        st.info(explanation)

        st.divider()
        if st.button("Reset — Check Another Patient"):
            st.rerun()
