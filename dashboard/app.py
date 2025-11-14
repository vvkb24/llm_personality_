import streamlit as st
import json
from components.radar_chart import plot_radar
from components.correlation_matrix import plot_corr

st.title("LLM Personality Dashboard")

uploaded = st.file_uploader("Upload results JSON", type="json")

if uploaded:
    data = json.load(uploaded)
    scores = data.get("scores", [])
    validity = data.get("validity", {})

    st.header("Big Five Radar Chart")
    plot_radar(validity)

    st.header("Correlation Matrix")
    plot_corr(validity)
