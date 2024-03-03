# from flask_cors import CORS
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# cors = CORS(app)

# Load the trained model (Pickle file)
model = pickle.load(open('heart.pkl', 'rb'))

# use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    print("------------Working--------------------")
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("------------Working Predict--------------------")

    int_features = [float(x) for x in request.form.values()] # Convert string inputs to float.
    features = [np.array(int_features)]  # Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)
    
    if output > 0.5:
        return render_template('index.html', pred='You have heart disease{}')
    else:
        return render_template('index.html', pred='You have no heart disease{}')

if __name__ == "__main__":
    app.run()
