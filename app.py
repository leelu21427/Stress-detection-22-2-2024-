import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,url_for,session,flash
import pickle
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import codecs
# Create Flask app
app = Flask(__name__)

model = pickle.load(open('model.pklz','rb'))
#model=joblib.load('model.pkl')
# Function to load the model
def load_model():
    global model
    model = pickle.load(open("model.pklz", "rb"))

# Route for home page
@app.route('/')
def home():
    return render_template("index.html")

# Route for prediction
@app.route('/predict', methods=["POST"])
def predict():
    # Ensure the model is loaded
    
    if model is None:
        load_model()
    
    # Get the features from the form
    float_features = [x for x in request.form.values()]
    float_features.remove('')
    features = [np.array(float_features)]
    
    # Make prediction
    prediction = model.predict(features)
    if prediction[0]==0:
        a="Positive Stress"
    elif prediction[0]==1:
        a="Low Stress"
    elif prediction[0]==2:
        a="Medium Stress"
    elif prediction[0]==3:
        a="High Stress"
    elif prediction[0]==4:
        a="Extreme Stress"
    
    # Return prediction to template
    
   
    return render_template("index.html", prediction_text="Stress level is: {}".format(a))
    
    #return "<html><body><h1>Stress level is:{}.format(a)</h1></body></html>"


# Main function
if __name__ =="__main__":
    app.run(debug=True,port=5001)

