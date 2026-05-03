import streamlit as st


def kpis(df):
    df["triple_risk_flag"] = (
        (df["hypertension"] == 1)
        & (df["heart_disease"] == 1)
        & (df["HbA1c_level"] >= 6.5)
    ).astype(int)

    df["prediabetic_flag"] = (
        (df["HbA1c_level"].between(5.7, 6.4))
        | (df["blood_glucose_level"].between(100, 125))
    ).astype(int)

    diabetic_percentage = df["diabetes"].mean() * 100
    non_diabetic_percentage = (1 - df["diabetes"].mean()) * 100
    risk_percentage = (len(df[df["triple_risk_flag"] == 1]) / len(df)) * 100
    prediabetic_percentage = (len(df[df["prediabetic_flag"] == 1]) / len(df)) * 100

    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        .kpi-box   { border-radius:10px; padding:20px; text-align:center; margin-top: 6px; box-shadow:2px 2px 8px rgba(0,0,0,0.1); }
        .kpi-icon  { font-size:24px; margin-bottom:6px; }
        .kpi-label { font-size:14px; font-weight:600; margin-bottom:8px; }
        .kpi-value { font-size:32px; font-weight:700; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#E8654C; color:#ffffff;">
                <div class="kpi-icon"><i class="fa-solid fa-droplet"></i></div>
                <div class="kpi-label">Diabetic</div>
                <div class="kpi-value">{diabetic_percentage:.1f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#4C9BE8; color:#ffffff;">
                <div class="kpi-icon"><i class="fa-solid fa-circle-check"></i></div>
                <div class="kpi-label">Non-Diabetic</div>
                <div class="kpi-value">{non_diabetic_percentage:.1f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#ffb74d; color:#ffffff;">
                <div class="kpi-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
                <div class="kpi-label">At Risk</div>
                <div class="kpi-value">{risk_percentage:.1f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#3f51b5; color:#ffffff;">
                <div class="kpi-icon"><i class="fa-solid fa-heart-pulse"></i></div>
                <div class="kpi-label">Pre-diabetic</div>
                <div class="kpi-value">{prediabetic_percentage:.1f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
