import streamlit as st
import numpy as np
import json
from pathlib import Path
from logic.context import ContextRetriever
from logic.velocity import VelocityEngine
from logic.slider import FrictionSlider

# --- Streamlit UI Config ---
st.set_page_config(page_title="SAVE Engine Simulator", layout="wide")
st.title("SAVE: Session-Aware Velocity Engine")

# --- Mock Data Simulation ---
# Load personas from personas.json
personas_path = Path(__file__).parent / "data" / "personas.json"
with open(personas_path, "r") as f:
    personas = json.load(f)

# Extract persona options for sidebar
persona_options = {persona["label"]: persona for persona in personas}

# --- Sidebar: User Persona & Attack Toggles ---
st.sidebar.header("User Context")
persona_choice = st.sidebar.selectbox("Active Persona", list(persona_options.keys()))
selected_persona = persona_options[persona_choice]
is_hijacked = st.sidebar.toggle("🚨 Simulate Session Hijack", value=False)

# --- Core Logic Execution ---
# Map JSON structure to expected format
student_persona = {
    "id": selected_persona["persona_id"],
    "label": selected_persona["label"],
    "avg_vel": selected_persona["behavioral_traits"]["avg_daily_velocity"],
    "base_t": selected_persona["default_threshold"],
    "weights": selected_persona["behavioral_traits"]["purpose_weights"]
}

retriever = ContextRetriever({"behavioral_traits": {"purpose_weights": student_persona["weights"]}})
velocity = VelocityEngine(student_persona["id"], student_persona["avg_vel"])
slider = FrictionSlider(base_threshold=student_persona["base_t"])

st.write("### Live Transaction Monitoring")
amount = st.slider("Transaction Amount (₱)", 10, 20000, 500)
purposes = list(student_persona["weights"].keys())
purpose = st.selectbox("Purpose", purposes)

if st.button("Authorize Transaction"):
    # Simulate Preamble: Hijacked sessions skip the balance check
    preamble_events = ["app_nav"] if not is_hijacked else []
    preamble_score = retriever.validate_human_preamble(preamble_events)
    
    # Calculate Signals
    drift = retriever.calculate_purpose_drift(purpose)
    _, vel_penalty = velocity.check_velocity()
    p_model = 0.85 if is_hijacked else 0.05  # Base ML score
    
    # Get Friction
    friction = slider.calculate_friction(p_model, drift, preamble_score, vel_penalty)
    
    # --- Visualization of Results ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Friction Score", friction)
        if friction > 0.7:
            st.error("ACTION: BLOCK & FACE-ID REQUIRED")
        else:
            st.success("ACTION: AUTO-APPROVE")
            
    with col2:
        # Show why the score is what it is
        st.write("**Risk Analysis:**")
        st.progress(drift, text=f"Purpose Drift: {drift}")
        st.progress(preamble_score, text=f"Preamble Anomaly: {preamble_score}")