from flask import Flask, request, jsonify
import joblib
import numpy as np

print("hello world");

app = Flask(__name__)
model = joblib.load("mnist_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=["GET"])
def home():
    return "MNIST Logistic Regression API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # fetch the json object from the POST request
    # features = scaler.transform(np.array([data['pixels']])) # extract the features
    features = np.array([data['pixels']]) # extract the features
    prediction = model.predict(features)
    print(prediction)
    return jsonify({'prediction': int(prediction[0])});

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    # print("app running succesfully")
    # this one will probably work.

def writing_the_fun():
    pass

