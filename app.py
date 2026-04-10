from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('./Model/api_predictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        try:
            # Get input values from form
            features = [
                float(request.form["PM10"]),
                float(request.form["NO"]),
                float(request.form["NO2"]),
                float(request.form["NOx"]),
                float(request.form["NH3"]),
                float(request.form["CO"]),
                float(request.form["SO2"]),
                float(request.form["O3"]),
                float(request.form["Benzene"]),
                float(request.form["Toluene"])
            ]

            # Convert to numpy array
            final_features = np.array([features])

            # Prediction
            prediction = model.predict(final_features)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)