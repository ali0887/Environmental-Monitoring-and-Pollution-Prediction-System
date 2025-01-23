import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib

class DataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and combine all CSV files from the data directory"""
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        dataframes = []
        
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
        
        return combined_df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Forward fill for missing values
        df = df.ffill()
        # If any remaining NaN at the start, backward fill
        df = df.bfill()
        return df
    
    def remove_outliers(self, df, columns=['temperature', 'humidity', 'wind_speed', 'air_quality_us']):
        """Remove outliers using IQR method"""
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
    
    def create_sequences(self, data, sequence_length=24):
        """Create sequences for LSTM model"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequence = data[i:(i + sequence_length)]
            target = data[i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, sequence_length=24, test_split=0.2):
        """Prepare data for training"""
        # Load and preprocess data
        df = self.load_data()
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        
        # Select features for modeling
        features = ['temperature', 'humidity', 'wind_speed', 'air_quality_us']
        data = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, sequence_length)
        
        # Split into training and testing sets
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Save the scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        return X_train, X_test, y_train, y_test, features

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, features = preprocessor.prepare_data()
    print("Data preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Features used: {features}")
