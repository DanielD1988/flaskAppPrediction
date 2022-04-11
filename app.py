"""
Name Daniel Dinelli
Student Number C00242741
Looked at this example https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf
"""

from os import environ
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request,render_template
import json
app = Flask(__name__)

model = None
"""
This dictionary is used to give back a string response back to the phone user
"""
thisdict = {
  "0": "Lower Back Pain",
  "1": "Plantar Fasciitis",
  "2": "Shin Splints",
  "3": "Sprains",
  "4": "Strains"
}
"""
This function loads the decsion tree classfier model from the pickle file
"""
def load_model():
    global model
    with open('decsionTree_model.pkl', 'rb') as f:
        model = pickle.load(f)

"""
This function used for testing purposes
"""
#@app.route('/')
#def home_endpoint():
#    load_model()
#    return "hello"
    #return render_template("index.html")

"""
This function takes in a post request with values sent from the mobile device that relate to the decsion tree model
the model than classifies the data passed as a number that number represents a injury type in the dictionary
it then sends back a response of the type of injury
"""
@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        prediction = model.predict([data])  # runs globally loaded model on the data
        pred = thisdict[str(prediction[0])]
    return pred

"""
This is the entry point of the app
"""
if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=8080)


