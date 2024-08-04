from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from datetime import datetime
from car_data_prep import prepare_data, get_columns, get_unique_values

app = Flask(__name__)

# Load the trained model and scaler
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    columns = get_columns()
    unique_values = get_unique_values()
    current_year = datetime.now().year
    return render_template('index.html', columns=columns, unique_values=unique_values, form_data={}, current_year=current_year)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    form_data = request.form.to_dict()
    
    # Ensure dates are in the correct format
    if 'Repub_date' in form_data and form_data['Repub_date']:
        form_data['Repub_date'] = pd.to_datetime(form_data['Repub_date'], format='%d/%m/%Y', errors='coerce')
    if 'Cre_date' in form_data and form_data['Cre_date']:
        form_data['Cre_date'] = pd.to_datetime(form_data['Cre_date'], format='%d/%m/%Y', errors='coerce')
    if 'Test' in form_data and form_data['Test']:
        form_data['Test'] = pd.to_datetime(form_data['Test'], format='%d/%m/%Y', errors='coerce')
    
    # Convert data to DataFrame
    df = pd.DataFrame([form_data])
    
    # Prepare the data for prediction
    df_prepared = prepare_data(df)
    
    # Check for NaN values
    if df_prepared.isnull().values.any():
        return jsonify({'error': f'Prepared data contains NaN values: {df_prepared.isnull().sum().to_dict()}'})
    
    # Ensure the features match the training features
    missing_cols = set(scaler.feature_names_in_) - set(df_prepared.columns)
    for col in missing_cols:
        df_prepared[col] = 0
    
    df_prepared = df_prepared[scaler.feature_names_in_]
    
    df_scaled = scaler.transform(df_prepared)
    
    # Predict the price
    prediction = model.predict(df_scaled)[0]
    
    # Ensure the predicted price is not negative
    prediction = max(prediction, 0)
    
    columns = get_columns()
    unique_values = get_unique_values()
    current_year = datetime.now().year
    return render_template('index.html', columns=columns, unique_values=unique_values, prediction_text=f'The predicted price is: {prediction:.2f}', form_data=form_data, current_year=current_year)

if __name__ == '__main__':
    app.run(debug=True)
