from flask import Flask, render_template, request, jsonify
import sys
import os
import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_service import ApiService
from src.models.data_models import PredictionRequest

app = Flask(__name__)
api_service = ApiService()

# Home route
@app.route('/')
def home():
    # Get available symbols for the dropdown
    symbol_list = api_service.get_available_symbols()
    return render_template('index.html', symbols=symbol_list.symbols)

# API status endpoint
@app.route('/api/status')
def api_status():
    status = api_service.get_api_status()
    return jsonify(status.dict())

# Available symbols endpoint
@app.route('/api/symbols')
def get_symbols():
    symbols = api_service.get_available_symbols()
    return jsonify(symbols.dict())

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    timeframe = request.form.get('timeframe', 'short_term')
    include_confidence = True if request.form.get('include_confidence') == 'on' else False
    
    # Create prediction request
    prediction_request = PredictionRequest(
        symbol=symbol,
        timeframe=timeframe,
        include_confidence=include_confidence
    )
    
    try:
        # Get prediction
        prediction_response = api_service.predict(prediction_request)
        return render_template('result.html', result=prediction_response)
    except ValueError as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
