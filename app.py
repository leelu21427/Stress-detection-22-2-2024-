'''import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import joblib
#Create flask app
app=Flask(__name__)

#Load the pickle model
model=pickle.load(open('model.pkl','rb'))
#model = joblib.load("model.joblib")
#route
@app.route('/',methods=["GET","POST"])
#Home page
def Home():
    return render_template("index.html")
@app.route('/predicts',methods=["POST"])
def predicts():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    #print(features)
    prediction=model.predict(features)
   
    return render_template("index.html",prediction_text="Stress level is: {}".format(prediction))

#main function
if __name__=="__main__":
    app.run(debug=True,port=5003)'''
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('model.pkl','rb'))

# Function to load the model
def load_model():
    global model
    model = pickle.load(open("model.pkl", "rb"))

# Route for home page
@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

# Route for prediction
@app.route('/predict', methods=["POST"])
def predict():
    # Ensure the model is loaded
    if model is None:
        load_model()
    
    # Get the features from the form
    float_features = [float(x) for x in request.form.values()]
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

# Main function
if __name__ == "__main__":
    app.run(debug=True,port=5001)
