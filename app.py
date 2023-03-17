from joblib import load
from flask import Flask, request, app, jsonify, url_for,render_template
import numpy as np
import pandas as pd

app= Flask(__name__)

# Load the model
model= load('detectorModel.joblib')
#loading my scaler
scaler= load('scaler.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():
    data=request.json['data']
    n_data= np.array(list(data.values())).reshape(1,-1)
    n_data= scaler.transform(n_data)
    prediction= model.predict(n_data)
    awnser= "no-fraud"
    if prediction[0]==1:
        awnser= "A fraudulet act occurs"
    return jsonify(awnser)

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    final_input= scaler.transform(np.array(data).reshape(1,-1))
    # print(final_input)
    prediction=model.predict(final_input)
    awnser= 'No Fradulent Activty on this card'
    if prediction[0]==1:
        awnser= 'A Fraudulent Act Occured!!'
    # print(jsonify(awnser))
    return render_template("home.html",prediction_text= awnser)

if __name__=="__main__":
    app.run(debug=True)