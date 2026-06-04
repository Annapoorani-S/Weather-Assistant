import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Weather Assistant",
    page_icon="🌦️",
    layout="centered"
)

# Load model
model = joblib.load("model.pkl")

st.title("🌦️ Weather Assistant")
st.markdown(
    "Predict weather conditions and get smart recommendations based on weather patterns."
)

st.divider()

# Input Section
col1, col2 = st.columns(2)

with col1:
    precipitation = st.number_input("🌧️ Precipitation", min_value=0.0)
    temp_max = st.number_input("🌡️ Maximum Temperature")

with col2:
    temp_min = st.number_input("❄️ Minimum Temperature")
    wind = st.number_input("💨 Wind Speed", min_value=0.0)

year = st.number_input("📅 Year", value=2026)

col3, col4 = st.columns(2)

with col3:
    month = st.number_input("Month", min_value=1, max_value=12, value=1)

with col4:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

temp_diff = temp_max - temp_min

if st.button("🔮 Predict Weather", use_container_width=True):

    features = np.array([[
        precipitation,
        temp_max,
        temp_min,
        wind,
        year,
        month,
        day,
        temp_diff
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.subheader("🤖 Weather Assistant")

    if prediction == "sun":
        st.success("☀️ Sunny")
        st.info("""
• Great day for outdoor activities
• Wear sunglasses
• Stay hydrated
• Good conditions for travel
""")

    elif prediction == "rain":
        st.success("🌧️ Rainy")
        st.warning("""
• Carry an umbrella
• Wear waterproof footwear
• Roads may be slippery
• Check traffic conditions
""")

    elif prediction == "drizzle":
        st.success("🌦️ Drizzle")
        st.info("""
• Light rain expected
• Carry a small umbrella
• Roads may be wet
• Good weather for indoor activities
""")

    elif prediction == "fog":
        st.success("🌫️ Foggy")
        st.warning("""
• Drive carefully
• Low visibility expected
• Allow extra travel time
• Use fog lights if driving
""")

    elif prediction == "snow":
        st.success("❄️ Snowy")
        st.warning("""
• Wear warm clothing
• Travel cautiously
• Watch for icy surfaces
• Keep emergency supplies ready
""")

    st.subheader("📊 Prediction Confidence")

    for weather, prob in zip(model.classes_, probabilities):
        st.write(f"{weather.capitalize()}: {prob*100:.2f}%")