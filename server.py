# This is a Flask application that forecasts disease cases based on historical data.
# pip install Flask
# pip install panda
# pip install numpy
# pip install matplotlib
# pip install seaborn
# pip install plotly_express
# pip install Prophet
# pip install itertools
# pip install statsmodels
# pip install sklearn
# pip install scikit-learn
# pip install openpyxl
# pip install requests

# Run this command in the terminal
# python main.py
#https://ploi.io/
#https://www.digitalocean.com/

from flask import Flask, request, jsonify

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from prophet import Prophet
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

app = Flask(__name__)

def adf_test(cases):
    result = adfuller(cases)
    labels = ['Test parameters', 'p-value', '#Lags Used', 'Dataset observations']
    for value, label in zip(result, labels):
        print(f"{label} : {value}")
    if result[1] <= 0.05:
        print("Dataset is stationary")
    else:
        print("Dataset is non-stationary")

def calculate_thresholds(data):
    mean_values = data.mean(axis=0)
    std_values = data.std(axis=0)
    alert_threshold = mean_values + std_values
    epidemic_threshold = mean_values + 1.65 * std_values
    return mean_values, std_values, alert_threshold, epidemic_threshold

def forecast_disease(disease, data):
    mp_cases_confirmed_excel = pd.read_excel(f'{disease}.xlsx')
    mp_cases_confirmed_excel = mp_cases_confirmed_excel.set_index(['Year', 'Morbidity_Week'])

    mp_cases_confirmed_threshold = pd.concat([mp_cases_confirmed_excel], sort=False).fillna(0)
    daily_country_cases_threshold = mp_cases_confirmed_threshold.groupby(['Morbidity_Week', 'Year']).size().reset_index(name='Total_cases')
    daily_country_cases_threshold = daily_country_cases_threshold.sort_values(['Year', 'Morbidity_Week'])

    df_received = pd.DataFrame(data)
    df_received = df_received.set_index(['Year', 'Morbidity_Week'])
    
    mp_cases_confirmed = pd.concat([mp_cases_confirmed_excel, df_received], sort=False).fillna(0)
    
    daily_country_cases = mp_cases_confirmed.groupby(['Morbidity_Week', 'Year']).size().reset_index(name='Total_cases')
    daily_country_cases = daily_country_cases.sort_values(['Year', 'Morbidity_Week'])
    
    daily_country_cases['Date'] = pd.to_datetime(daily_country_cases['Year'].astype(str) +
                                                 daily_country_cases['Morbidity_Week'].astype(str) + '1',
                                                 format='%Y%W%w')
    daily_country_cases.set_index('Date', inplace=True)
    
    adf_test(daily_country_cases['Total_cases'])
    
    train = daily_country_cases[:int(0.8 * len(daily_country_cases))]
    test = daily_country_cases[int(0.8 * len(daily_country_cases)):]
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_pdq = None
    
    for param in pdq:
        try:
            model = sm.tsa.arima.ARIMA(daily_country_cases['Total_cases'], order=param)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue
    
    model_arima = sm.tsa.arima.ARIMA(daily_country_cases['Total_cases'], order=best_pdq)
    model_arima_fit = model_arima.fit()
    
    forecast_arima = model_arima_fit.forecast(steps=4)
    forecast_df = pd.DataFrame({'Date': pd.date_range(start=test.index[-1], periods=4, freq='W'),
                                'Forecast': forecast_arima})
    
    combined_data = pd.concat([daily_country_cases, forecast_df], ignore_index=True)

   # Pivot data to compute thresholds
    pivot_data = daily_country_cases_threshold.pivot(index='Year', columns='Morbidity_Week', values='Total_cases').fillna(0)

    # Calculate thresholds
    mean_values, std_values, alert_threshold, epidemic_threshold = calculate_thresholds(pivot_data)

    # Add thresholds back to the combined data
    combined_data['alert_threshold'] = alert_threshold
    combined_data['epidemic_threshold'] = epidemic_threshold

    return combined_data.to_json(orient='records')

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()['data']
        disease = request.get_json()['disease'].upper()
        forecast_data = forecast_disease(disease, data)
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecastalldisease', methods=['POST'])
def forecastalldisease():
    try:
        diseases = request.get_json()['diseases']
        all_forecasts = {disease: forecast_disease(disease, data) for disease, data in diseases.items()}
        return jsonify(all_forecasts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')  # Add this route
def root():
    return jsonify({'status': 'ok'}), 200
    
@app.route('/health')
def health_check():
    return jsonify({'status': 'Health Ok'}), 200
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080)) # Get port from environment or default to 8080
    app.run(host='0.0.0.0', port=port, debug=False) #Disable debug for production.
    # app.run(debug=True)
