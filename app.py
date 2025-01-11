#import required libraries
from flask import Flask, request, jsonify
import joblib
import numpy as np

#Create Flask API instance
app = Flask(__name__)

#Load the ML model from the joblib file
model = joblib.load('model\model.joblib')

#Default route on launching the Flask API
@app.route('/', methods=['GET'])
def default():
    return 'Flask API running successfully.'

#API POST method/route for prediction of given inputs
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array(data['input']).reshape(1,-1))
    return jsonify({'prediction': prediction.tolist()})

#Starting the Flask API
if __name__ == '__main__':
    app.run(debug=True)