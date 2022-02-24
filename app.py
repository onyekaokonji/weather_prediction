import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template, url_for, flash, redirect

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    pressure = request.form.get('pressure')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    wind_speed = request.form.get('wind speed')
    wind_direction = request.form.get('wind direction')
    year = request.form.get('year')
    month = request.form.get('month')
    day = request.form.get('day')
    hour = request.form.get('hour')
    average_humidity = request.form.get('average humidity')
    average_pressure = request.form.get('average pressure')
    average_temperature = request.form.get('average temperature')
    average_wind_direction = request.form.get('average wind direction')
    average_wind_speed = request.form.get('average wind speed')
    inputs = [[pressure, humidity, temperature, wind_speed, wind_direction,
                                year, month, day, hour, average_humidity, average_pressure, average_temperature,
                                average_wind_direction, average_wind_speed]]
    final_features = np.array(inputs)
    test_data = pd.DataFrame(final_features, columns = ['pressure', 'temperature', 'humidity',
                                                        'wind_speed', 'wind_direction', 'year',
                                                        'month', 'day', 'hour', 'average_humidity',
                                                        'avergae_pressure','average_temperature',
                                                        'average_wind_direction','average_wind_speed'])

    model = pickle.load(open('model.pkl', 'rb'))

    prediction = model.predict(test_data)

    
    return render_template('index.html', prediction_text='Weather Condition: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
