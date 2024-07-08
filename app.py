from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load the model
model = joblib.load('advertising_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tv_budget = float(data['tv_budget'])
    radio_budget = float(data['radio_budget'])
    newspaper_budget = float(data['newspaper_budget'])

    # Prepare the input for prediction
    input_features = np.array([1, tv_budget, radio_budget, newspaper_budget]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)[0]

    # Generate and save the graph
    df = pd.read_csv("Advertising Budget and Sales.csv", names=['id', 'TV_Budget', 'Radio_Budget', 'NP_Budget', 'Sales'])
    df = df.drop(columns=['id'])
    df = df.drop(index=df.index[0])
    df[['TV_Budget', 'Radio_Budget', 'NP_Budget', 'Sales']] = df[['TV_Budget', 'Radio_Budget', 'NP_Budget', 'Sales']].astype(float)

    plt.figure(figsize=(10, 6))
    sns.pairplot(df, x_vars=['TV_Budget', 'Radio_Budget', 'NP_Budget'], y_vars=['Sales'], height=5)
    plt.savefig('static/pairplot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(data=df.corr(), annot=True, linewidths=2, linecolor='black', fmt='.1g', center=0.6)
    plt.savefig('static/heatmap.png')
    plt.close()

    return jsonify({'prediction': prediction})

@app.route('/static/<path:path>')
def send_static(path):
    return send_file(f'static/{path}')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
