import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# API endpoint
API_URL = "http://localhost:8000/predict"

# Get available models
try:
    response = requests.get("http://localhost:8000/")
    available_models = response.json()["available_models"]
except:
    available_models = ["2", "6", "12", "24"]  # Fallback if API is not running

st.set_page_config(page_title="Pollution Prediction Dashboard", layout="wide")

st.title("🌍 Air Quality Prediction Dashboard")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Weather Conditions")
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    wind_speed = st.number_input("Wind Speed (m/s)", value=5.0)
    air_quality = st.number_input("Current Air Quality Index (US)", value=50.0)
    
    # Add model selection
    model_type = st.selectbox(
        "Select Prediction Window (hours)",
        options=available_models,
        help="Choose how many hours of historical data to use for prediction"
    )

    if st.button("Predict"):
        # Prepare the data
        data = {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "air_quality_us": air_quality,
            "model_type": model_type
        }

        try:
            # Make prediction
            response = requests.post(API_URL, json=data)
            prediction = response.json()

            with col2:
                st.subheader(f"Prediction Results (Using {model_type}-hour window)")
                
                # Create metrics
                col2_1, col2_2 = st.columns(2)
                
                with col2_1:
                    st.metric(
                        label="Predicted Temperature",
                        value=f"{prediction['predicted_temperature']:.1f}°C",
                        delta=f"{prediction['predicted_temperature'] - temperature:.1f}°C"
                    )
                    st.metric(
                        label="Predicted Wind Speed",
                        value=f"{prediction['predicted_wind_speed']:.1f} m/s",
                        delta=f"{prediction['predicted_wind_speed'] - wind_speed:.1f} m/s"
                    )
                
                with col2_2:
                    st.metric(
                        label="Predicted Humidity",
                        value=f"{prediction['predicted_humidity']:.1f}%",
                        delta=f"{prediction['predicted_humidity'] - humidity:.1f}%"
                    )
                    st.metric(
                        label="Predicted AQI",
                        value=f"{prediction['predicted_air_quality']:.1f}",
                        delta=f"{prediction['predicted_air_quality'] - air_quality:.1f}"
                    )
                
                # Create AQI gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction['predicted_air_quality'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Air Quality Index"},
                    gauge = {
                        'axis': {'range': [0, 300]},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [51, 100], 'color': "yellow"},
                            {'range': [101, 150], 'color': "orange"},
                            {'range': [151, 200], 'color': "red"},
                            {'range': [201, 300], 'color': "purple"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction['predicted_air_quality']
                        }
                    }
                ))
                
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Display historical data
st.subheader("Historical Data")
try:
    # Load the most recent data file
    df = pd.read_csv("data/weather_data_20241214_190104.csv")  # Update this to load your actual data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['air_quality_us'],
                            mode='lines+markers', name='AQI'))
    
    fig.update_layout(title='Historical Air Quality Index',
                     xaxis_title='Time',
                     yaxis_title='AQI (US)')
    
    st.plotly_chart(fig)
    
except Exception as e:
    st.error(f"Error loading historical data: {str(e)}")

# Add information about AQI levels
st.subheader("AQI Levels Information")
aqi_info = """
- 0-50: Good (Green) - Air quality is satisfactory
- 51-100: Moderate (Yellow) - Acceptable air quality
- 101-150: Unhealthy for Sensitive Groups (Orange)
- 151-200: Unhealthy (Red) - Everyone may experience health effects
- 201-300: Very Unhealthy (Purple) - Health alert
"""
st.markdown(aqi_info)
