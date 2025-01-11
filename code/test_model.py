#import the required libraries
import joblib
import os

#Load the model using joblib
model_path = os.path.join('model', 'model.joblib')
my_model = joblib.load(model_path)

#Test the prediction for two values
def test_predict():
    assert my_model.predict([[6.8,2.8,4.8,1.4]]) == 1
    assert my_model.predict([[6.5,3.0,5.2,2.0]]) == 0
