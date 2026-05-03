import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main_chart(df):

    feature_options = {
        "Age (years)": "age",
        "HbA1c Level (%)": "HbA1c_level",
        "Blood Glucose Level (mg/dL)": "blood_glucose_level",
    }

    selected_label = st.selectbox("Select Feature", list(feature_options.keys()))
    selected_col = feature_options[selected_label]

    no_dm = df[df["diabetes"] == 0][selected_col]
    dm = df[df["diabetes"] == 1][selected_col]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Distribution (Histogram + KDE)", "Box")
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=no_dm,
            name="No Diabetes",
            marker_color="#4C9BE8",
            opacity=0.6,
            histnorm="probability density",
            nbinsx=35,
            legendgroup="no_dm",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=dm,
            name="Diabetes",
            marker_color="#E8654C",
            opacity=0.6,
            histnorm="probability density",
            nbinsx=35,
            legendgroup="dm",
        ),
        row=1,
        col=1,
    )

    # Box Plot
    fig.add_trace(
        go.Box(
            y=no_dm,
            name="No Diabetes",
            marker_color="#4C9BE8",
            boxmean=True,
            legendgroup="no_dm",
            showlegend=False,  # ✅ hides from top-right legend
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Box(
            y=dm,
            name="Diabetes",
            marker_color="#E8654C",
            boxmean=True,
            legendgroup="dm",
            showlegend=False,  # ✅ hides from top-right legend
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{selected_label} Distribution by Diabetes Status",
        barmode="overlay",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    st.plotly_chart(fig, width="stretch")

    # Key stats below chart

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#4C9BE8; color:#ffffff;">
                <div class="kpi-label">No Diabetes Mean</div>
                <div class="kpi-value">{no_dm.mean():.2f}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#E8654C; color:#ffffff;">
                <div class="kpi-label">Diabetes Mean</div>
                <div class="kpi-value">{dm.mean():.2f}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-box" style="background-color:#3f51b5; color:#ffffff;">
                <div class="kpi-label">Δ Difference</div>
                <div class="kpi-value">{dm.mean() - no_dm.mean():+.2f}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
