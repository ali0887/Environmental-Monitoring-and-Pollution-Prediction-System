import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from data_preprocessing import DataPreprocessor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import itertools

def create_model(sequence_length, n_features, units=50, dropout=0.2, learning_rate=0.001):
    """Create LSTM model"""
    inputs = tf.keras.Input(shape=(sequence_length, n_features))
    x = LSTM(units, return_sequences=True)(inputs)
    x = Dropout(dropout)(x)
    x = LSTM(units)(x)
    x = Dropout(dropout)(x)
    outputs = Dense(n_features)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def get_run_name(params):
    """Create run name from parameters"""
    return f"experiment_{params['sequence_length']}_{params['lstm_units']}_{params['dropout']}_{params['learning_rate']}_{params['epochs']}_{params['batch_size']}"

def train_model(params, sequence_best_rmse):
    """Train model with given parameters and log metrics with MLflow"""
    mlflow.set_experiment("MLOPss_Pollutionss_Predictionsss")
    
    with mlflow.start_run(run_name=get_run_name(params)):
        # Log parameters
        mlflow.log_params(params)
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, features = preprocessor.prepare_data(
            sequence_length=params['sequence_length']
        )
        
        # Create and train model
        model = create_model(
            sequence_length=params['sequence_length'],
            n_features=len(features),
            units=params['lstm_units'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Log validation metrics from history
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        mlflow.log_metric("val_mse", val_loss)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", np.sqrt(val_loss))
        
        # Log model
        mlflow.keras.log_model(model, "model")
        
        # If this is the best model for this sequence length, save it
        if rmse < sequence_best_rmse:
            os.makedirs('models', exist_ok=True)
            model_path = f'models/best_model_seq_{params["sequence_length"]}.keras'
            model.save(model_path)
            return model, rmse, mae, True
        
        return model, rmse, mae, False

def grid_search():
    """Perform grid search over hyperparameters"""
    # Define parameter grid
    param_grid = {
        'sequence_length': [2, 6, 12, 24],
        'lstm_units': [16, 32, 64, 128],
        'dropout': [0.1, 0.2],
        'learning_rate': [0.001],
        'epochs': [30, 50],
        'batch_size': [4, 8, 16]
    }
    
    # Get all combinations of parameters
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Dictionary to keep track of best RMSE for each sequence length
    sequence_best_rmse = {seq_len: float('inf') for seq_len in param_grid['sequence_length']}
    
    # Dictionary to keep track of best parameters for each sequence length
    sequence_best_params = {seq_len: None for seq_len in param_grid['sequence_length']}
    
    total_combinations = len(param_combinations)
    print(f"Total number of combinations to try: {total_combinations}")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\nTraining combination {i}/{total_combinations}")
        print("Parameters:", params)
        
        try:
            current_seq_len = params['sequence_length']
            current_best_rmse = sequence_best_rmse[current_seq_len]
            
            _, rmse, _, is_best = train_model(params, current_best_rmse)
            
            if is_best:
                sequence_best_rmse[current_seq_len] = rmse
                sequence_best_params[current_seq_len] = params.copy()
                print(f"New best RMSE for sequence length {current_seq_len}: {rmse:.4f}")
        except Exception as e:
            print(f"Error training with parameters {params}: {str(e)}")
            continue
    
    print("\nGrid Search completed!")
    print("\nBest parameters and RMSE for each sequence length:")
    for seq_len in param_grid['sequence_length']:
        print(f"\nSequence Length {seq_len}:")
        print(f"Best parameters: {sequence_best_params[seq_len]}")
        print(f"Best RMSE: {sequence_best_rmse[seq_len]:.4f}")
    
    # Find overall best sequence length
    best_seq_len = min(sequence_best_rmse.items(), key=lambda x: x[1])[0]
    print(f"\nOverall best sequence length: {best_seq_len}")
    print(f"Overall best RMSE: {sequence_best_rmse[best_seq_len]:.4f}")
    
    return sequence_best_params, sequence_best_rmse

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    best_params_dict, best_rmse_dict = grid_search()
    
    print("\nFinal models have been saved in the 'models' directory:")
    for seq_len in best_params_dict.keys():
        print(f"- models/best_model_seq_{seq_len}.keras")
