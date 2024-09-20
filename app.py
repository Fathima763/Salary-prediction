import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model(4).pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def predict():

    if request.method == 'POST':
        # Get input values from the form
        age = float(request.form['age'])  # Make sure the form name matches
        gender = request.form['gender']
        education = request.form['Education_Level']
        job_title = request.form['job_title']
        experience = float(request.form['experience'])

        gender_encoded= {'Male': 0, ' Female': 1}[gender_encoded]  # Adjust as needed
        education_encoded = {"Bachelor's Degree": 1, "High School": 2,"Master's Degree": 4,'PhD': 5}[education_encoded]  # Adjust based on your crops
        job_title_encoded = {'Data Analyst': 0, 'Director': 1, 'Director of Marketing': 2, 'Financial Manager': 3, 'Marketing Manager':4 ,'Sales Associate': 5, 'Sales Executive':6,'Senior Manager':7,'Software Engineer':8}[job_title_encoded]  # Adjust based on soil types


        # Create input array for prediction
        input_data = [[age, gender_encoded,  education_encoded, job_title_encoded, experience]]

        # Scale the input data using the pre-fitted scaler
        # input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run()


