@echo off
cd /d "G:\Semester 7\MLOPs\course-project-ali0887"

:: Run the Python script to fetch weather data
python fetch_weather_data.py

:: Add new data to DVC
python -m dvc add data/

:: Commit changes to git
git add .gitignore data.dvc
git commit -m "Updated weather data at %DATE% %TIME%"
git push origin main

:: Push changes to DVC remote
python -m dvc push
