import pickle
import numpy as np
from flask import Flask, request, render_template

# Load trained model
with open("fish_species_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the LabelEncoder used during training
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        features = [
            float(request.form['Weight']),
            float(request.form['Length1']),
            float(request.form['Length2']),
            float(request.form['Length3']),
            float(request.form['Height']),
            float(request.form['Width'])
        ]

        # Convert input to numpy array and reshape for prediction
        input_features = np.array([features])

        # Make prediction (numeric label)
        prediction_numeric = model.predict(input_features)[0]

        # Convert numeric prediction to actual species name
        prediction_species = label_encoder.inverse_transform([prediction_numeric])[0]

        return render_template('index.html', prediction=f"Predicted Species: {prediction_species}")

    except Exception as e:
        return render_template('index.html', prediction="Error: Invalid input!")

if __name__ == "__main__":
    app.run(debug=True)
