import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv('IQAIR_API_KEY')

# Create data directory if it doesn't exist
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

def fetch_weather_data(lat=33.6007, lon=73.0679):  # Default coordinates for Islamabad
    """
    Fetch weather data from IQAir API for given coordinates
    """
    base_url = 'http://api.airvisual.com/v2/nearest_city'
    params = {
        'lat': lat,
        'lon': lon,
        'key': API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_weather_data(data):
    """
    Save weather data to JSON and CSV files
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw JSON data
    json_filename = os.path.join(data_dir, f'weather_data_{timestamp}.json')
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Extract relevant weather information for CSV
    if data and 'data' in data:
        weather_data = data['data']
        weather_dict = {
            'timestamp': [datetime.now().isoformat()],
            'city': [weather_data.get('city')],
            'country': [weather_data.get('country')],
            'temperature': [weather_data.get('current', {}).get('weather', {}).get('tp')],
            'humidity': [weather_data.get('current', {}).get('weather', {}).get('hu')],
            'wind_speed': [weather_data.get('current', {}).get('weather', {}).get('ws')],
            'air_quality_us': [weather_data.get('current', {}).get('pollution', {}).get('aqius')],
            'air_quality_cn': [weather_data.get('current', {}).get('pollution', {}).get('aqicn')]
        }
        
        df = pd.DataFrame(weather_dict)
        csv_filename = os.path.join(data_dir, f'weather_data_{timestamp}.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"Data saved successfully:\nJSON: {json_filename}\nCSV: {csv_filename}")
    else:
        print("No data to save")

def main():
    # Fetch weather data
    weather_data = fetch_weather_data()
    
    if weather_data:
        # Save the data
        save_weather_data(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()
