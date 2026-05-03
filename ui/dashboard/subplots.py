import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots


def gender_age_heatmap(df):
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"],
    )
    gender_age = (
        df.groupby(["gender", "age_group"], observed=True)["diabetes"].mean() * 100
    )
    gender_age_pivot = gender_age.unstack(level="age_group")

    fig = go.Figure(
        data=go.Heatmap(
            z=gender_age_pivot.values,
            x=gender_age_pivot.columns.astype(str).tolist(),
            y=gender_age_pivot.index.tolist(),
            colorscale="Blues",
            text=gender_age_pivot.values.round(1),
            texttemplate="%{text}%",
            textfont=dict(size=12),
            colorbar=dict(title="Diabetes Rate (%)"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Diabetes Rate (%) by Gender and Age Group",
        xaxis_title="Age Group",
        yaxis_title="Gender",
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333"),
    )

    st.plotly_chart(fig, width="stretch")


def risk_flags_chart(df):
    flags = ["high_hba1c_flag", "senior_flag", "cardio_risk_flag"]
    flag_labels = [
        "High HbA1c (≥6.6%)",
        "Senior (age ≥ 60)",
        "Cardio Risk (hypert. or heart dis.)",
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=flag_labels,
    )

    avg_diabetes_rate = df["diabetes"].mean() * 100

    for i, (flag, label) in enumerate(zip(flags, flag_labels), start=1):
        rates = df.groupby(flag)["diabetes"].mean() * 100
        counts = df.groupby(flag)["diabetes"].count()

        x_labels = [
            f"Flag=0 (n={counts.get(0, 0):,})",
            f"Flag=1 (n={counts.get(1, 0):,})",
        ]
        y_values = [rates.get(0, 0), rates.get(1, 0)]

        # Bars
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_values,
                marker_color=["#4C9BE8", "#E8654C"],
                text=[f"{v:.1f}%" for v in y_values],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=i,
        )

        # Average reference line
        fig.add_hline(
            y=avg_diabetes_rate,
            line_dash="dash",
            line_color="gray",
            line_width=1.2,
            row="all",
            col=i,
        )

    fig.update_layout(
        title="Diabetes Rate by Binary Risk Flags",
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333"),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%")

    st.plotly_chart(fig, width="stretch")


def bmi_diabetes_chart(df):
    bmi_order = ["Underweight", "Healthy Weight", "Overweight", "Obesity"]

    bmi_dm = (
        df.groupby("bmi_category", observed=True)["diabetes"].mean() * 100
    ).reindex(bmi_order)

    avg_diabetes_rate = df["diabetes"].mean() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bmi_dm.index.tolist(),
            y=bmi_dm.values,
            marker_color=["#4C9BE8", "#4C9BE8", "#FFA55A", "#E8654C"],
            text=[f"{v:.1f}%" for v in bmi_dm.values],
            textposition="outside",
            showlegend=False,
        )
    )

    # Average reference line 
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(bmi_order) - 0.5,
        y0=avg_diabetes_rate,
        y1=avg_diabetes_rate,
        line=dict(dash="dash", color="gray", width=1.2),
    )

    # Average label 
    fig.add_annotation(
        x=len(bmi_order) - 0.5,
        y=avg_diabetes_rate,
        text=f"Avg: {avg_diabetes_rate:.1f}%",
        showarrow=False,
        xanchor="left",
        font=dict(color="gray", size=11),
    )

    fig.update_layout(
        title="Diabetes Rate by BMI Category",
        xaxis_title="BMI Category",
        yaxis_title="Diabetes Rate (%)",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333"),
        xaxis=dict(tickangle=20),
    )

    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%")
    fig.update_xaxes(showgrid=False)

    st.plotly_chart(fig, width="stretch")
