from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import os
import mlflow
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Prometheus metrics
PREDICTIONS_COUNTER = Counter(
    'pollution_predictions_total',
    'Total number of pollution predictions made',
    ['model_type']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction',
    ['model_type'],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Add gauges for each prediction type
TEMPERATURE_GAUGE = Gauge('predicted_temperature', 'Last predicted temperature')
HUMIDITY_GAUGE = Gauge('predicted_humidity', 'Last predicted humidity')
WIND_SPEED_GAUGE = Gauge('predicted_wind_speed', 'Last predicted wind speed')
AIR_QUALITY_GAUGE = Gauge('predicted_air_quality', 'Last predicted air quality')

# Start Prometheus metrics server on a different port
start_http_server(8001)

app = FastAPI(title="Pollution Prediction API")

# Dictionary to store models
models = {}
scaler = None

class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    air_quality_us: float
    model_type: str  # Add model type selection

def load_models():
    global models, scaler
    try:
        # Try loading models from MLflow first
        experiment_name = "MLOPss_Pollutionss_Predictionsss"
        mlflow.set_tracking_uri("file:./mlruns")   # Updated path to use root directory
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is not None:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            # Get the best run for each sequence length
            for sequence_length in [2, 6, 12, 24]:
                sequence_runs = runs[runs["params.sequence_length"] == str(sequence_length)]
                if not sequence_runs.empty:
                    best_run = sequence_runs.sort_values("metrics.val_rmse").iloc[0]
                    run_id = best_run.run_id
                    model_path = f"mlruns/{experiment.experiment_id}/{run_id}/artifacts/model/data/model.keras"
                    models[str(sequence_length)] = tf.keras.models.load_model(model_path)
        
        # If no models loaded from MLflow, try loading from local models directory
        if not models:
            print("No models found in MLflow, trying local models directory...")
            model_dir = 'models'
            for sequence_length in [2, 6, 12, 24]:
                model_path = os.path.join(model_dir, f'best_model_seq_{sequence_length}.keras')
                if os.path.exists(model_path):
                    print(f"Loading model for sequence length {sequence_length} from {model_path}")
                    models[str(sequence_length)] = tf.keras.models.load_model(model_path)
        
        if not models:
            raise Exception("No models found in MLflow or local directory")
        
        # Load scaler
        scaler_path = os.path.join('models', 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            raise Exception("Scaler not found")
            
        print(f"Successfully loaded {len(models)} models. Available sequences: {list(models.keys())}")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
def read_root():
    return {"message": "Pollution Prediction API", "available_models": list(models.keys())}

@app.post("/predict")
async def predict(data: PredictionInput):
    logger.info(f"Received prediction request for model type: {data.model_type}")
    
    if data.model_type not in models:
        logger.error(f"Model type {data.model_type} not found. Available models: {list(models.keys())}")
        raise HTTPException(status_code=400, detail=f"Model type {data.model_type} not found. Available models: {list(models.keys())}")
    
    try:
        # Start timing the prediction
        start_time = time.time()
        
        # Prepare input data
        input_data = np.array([[
            data.temperature,
            data.humidity,
            data.wind_speed,
            data.air_quality_us
        ]])
        
        # Scale input data
        scaled_input = scaler.transform(input_data)
        
        # Reshape for LSTM based on sequence length
        sequence_length = int(data.model_type)
        scaled_input = np.tile(scaled_input, (1, sequence_length, 1))
        
        # Make prediction
        prediction = models[data.model_type].predict(scaled_input)
        
        # Inverse transform the prediction
        prediction = scaler.inverse_transform(prediction.reshape(1, -1))
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Update Prometheus metrics
        logger.info("Updating Prometheus metrics...")
        try:
            # Update counter and latency
            PREDICTIONS_COUNTER.labels(model_type=data.model_type).inc()
            PREDICTION_LATENCY.labels(model_type=data.model_type).observe(prediction_time)
            
            # Update gauges with latest predictions
            TEMPERATURE_GAUGE.set(float(prediction[0][0]))
            HUMIDITY_GAUGE.set(float(prediction[0][1]))
            WIND_SPEED_GAUGE.set(float(prediction[0][2]))
            AIR_QUALITY_GAUGE.set(float(prediction[0][3]))
            logger.info("Successfully updated Prometheus metrics")
        except Exception as metric_error:
            logger.error(f"Error updating metrics: {str(metric_error)}")
        
        response = {
            "predicted_temperature": float(prediction[0][0]),
            "predicted_humidity": float(prediction[0][1]),
            "predicted_wind_speed": float(prediction[0][2]),
            "predicted_air_quality": float(prediction[0][3]),
            "model_type": data.model_type,
            "prediction_time": prediction_time
        }
        
        logger.info(f"Prediction successful: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)