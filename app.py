from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/exoplanet_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on user input"""
    try:
        # Get data from the form
        features = []
        for feature in feature_names:
            value = float(request.form[feature])
            features.append(value)

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Prepare result
        result = {
            'prediction': 'Confirmed Exoplanet! 🪐' if prediction == 1 else 'False Positive ❌',
            'confidence': f"{max(probability) * 100:.2f}%",
            'probability_false': f"{probability[0] * 100:.2f}%",
            'probability_confirmed': f"{probability[1] * 100:.2f}%"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
