import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from engine import load_checkpoint, predict_and_reconstruct

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="POD-FCDNN Surrogate Model",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 POD-FCDNN Fluid Dynamics Surrogate Model")
st.markdown("Train and deploy a neural network surrogate model for rapid CFD prediction.")

# ------------------------------------------------------------
# CACHED MODEL LOAD (Loads once across app interactions)
# ------------------------------------------------------------
@st.cache_resource
def get_model(case_name: str):
    checkpoint_paths = {
        "Cavity": "checkpoints/cavity_checkpoint.pt",
        "Cylinder": "checkpoints/cylinder_checkpoint.pt",
        "Backward Facing Step": "checkpoints/bfs_checkpoint.pt",
        "NACA0012": "checkpoints/naca_checkpoint.pt"
    }
    return load_checkpoint(checkpoint_paths[case_name])

# ------------------------------------------------------------
# INPUT FORM (Prevents re-running on every slider move)
# ------------------------------------------------------------
with st.form("prediction_form"):
    case = st.selectbox(
        "Select Case",
        ["Cavity", "Cylinder", "Backward Facing Step", "NACA0012"]
    )

    if case == "NACA0012":
        param = st.slider("Angle of Attack (α)", min_value=-5.0, max_value=15.0, value=0.0, step=0.5)
    else:
        param = st.slider("Reynolds Number", min_value=100, max_value=10000, value=1000, step=100)

    predict_btn = st.form_submit_button("Predict Flow Field", use_container_width=True)

# ------------------------------------------------------------
# VECTORIZED PLOTTING FUNCTION
# ------------------------------------------------------------
def plot_field(x, y, values, title, colorscale="Viridis"):
    fig = go.Figure(
        go.Scattergl(  # WebGL for fast rendering of large point sets
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=4,
                color=values,
                colorscale=colorscale,
                showscale=True
            )
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ------------------------------------------------------------
# INFERENCE & DISPLAY
# ------------------------------------------------------------
if predict_btn:
    try:
        with st.spinner("Generating inference..."):
            trainer = get_model(case)
            result = predict_and_reconstruct(trainer, param)

            u = result["u"]
            v = result["v"]
            p = result["p"]
            xy = result["xy"]

            x_coords = xy[:, 0]
            y_coords = xy[:, 1]

        st.success(f"Prediction completed for {case}")

        # Render field visualizations using WebGL
        st.subheader("Pressure Field")
        st.plotly_chart(plot_field(x_coords, y_coords, p, f"{case} Pressure Field", "Viridis"), use_container_width=True)

        st.subheader("U Velocity")
        st.plotly_chart(plot_field(x_coords, y_coords, u, f"{case} U Velocity", "RdBu_r"), use_container_width=True)

        st.subheader("V Velocity")
        st.plotly_chart(plot_field(x_coords, y_coords, v, f"{case} V Velocity", "RdBu_r"), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
