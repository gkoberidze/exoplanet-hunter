from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd
from upload_handler import allowed_file, process_uploaded_csv
from explanations import (get_feature_explanation, get_prediction_insight,
                          get_exoplanet_facts, compare_to_solar_system)
from save_handler import save_prediction_to_history, load_prediction_history, clear_prediction_history, export_history_to_csv
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs('uploads', exist_ok=True)

# Load all available models
available_models = {
    'Best Model': 'models/best_model.pkl',
    'Random Forest Enhanced': 'models/random_forest_enhanced.pkl',
    'XGBoost Enhanced': 'models/xgboost_enhanced.pkl',
    'Neural Network Enhanced': 'models/neural_network_enhanced.pkl'
}

# Load default model and scaler
current_model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler_enhanced.pkl')
feature_names = joblib.load('models/best_feature_names.pkl')

# Load model comparison results
with open('models/enhanced_results.json', 'r') as f:
    model_results = json.load(f)


@app.route('/')
def home():
    """Render the main page"""

    # Get list of visualization images
    visualizations = []
    viz_files = [
        ('enhanced_model_comparison.png', 'Model Performance Comparison'),
        ('confusion_matrices_all.png', 'Confusion Matrices'),
        ('feature_importance_best.png', 'Feature Importance'),
        ('model_comparison.png', 'Initial Model Comparison')
    ]

    for filename, title in viz_files:
        if os.path.exists(f'static/{filename}'):
            visualizations.append({'filename': filename, 'title': title})

    return render_template('index.html',
                           features=feature_names,
                           models=list(available_models.keys()),
                           model_results=model_results,
                           visualizations=visualizations)


@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on user input"""
    try:
        # Get selected model
        selected_model_name = request.form.get('model', 'Best Model')

        # Load the selected model
        if selected_model_name in available_models:
            model = joblib.load(available_models[selected_model_name])
        else:
            model = current_model

        # Get feature values
        features = []
        feature_values_dict = {}
        for feature in feature_names:
            value = float(request.form[feature])
            features.append(value)
            feature_values_dict[feature] = value

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            }
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(),
                                             key=lambda x: x[1], reverse=True)[:5])

        # Get detailed insights
        insights = get_prediction_insight(
            prediction, probability, feature_values_dict, feature_importance)
        solar_system_comparison = compare_to_solar_system(feature_values_dict)

        # Prepare result
        result = {
            'prediction': 'Confirmed Exoplanet! 🪐' if prediction == 1 else 'False Positive ❌',
            'confidence': f"{max(probability) * 100:.2f}%",
            'probability_false': f"{probability[0] * 100:.2f}%",
            'probability_confirmed': f"{probability[1] * 100:.2f}%",
            'model_used': selected_model_name,
            'feature_importance': feature_importance,
            'input_values': feature_values_dict,
            'insights': insights,
            'solar_system_comparison': solar_system_comparison
        }

        # Save to history
        save_prediction_to_history(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload and batch predictions"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        # Get selected model
        selected_model_name = request.form.get('model', 'Best Model')

        # Load the selected model
        if selected_model_name in available_models:
            model = joblib.load(available_models[selected_model_name])
        else:
            model = current_model

        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the CSV
        result = process_uploaded_csv(filepath, scaler, model, feature_names)

        # Clean up
        os.remove(filepath)

        if 'error' in result:
            return jsonify(result), 400

        result['model_used'] = selected_model_name
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing upload: {str(e)}'}), 500


@app.route('/download_results', methods=['POST'])
def download_results():
    """Download prediction results as CSV"""
    try:
        data = request.json
        df = pd.DataFrame(data['results'])

        # Create CSV in memory
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='exoplanet_predictions.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def get_history():
    """Get prediction history"""
    history = load_prediction_history()
    return jsonify({'history': history})


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    clear_prediction_history()
    return jsonify({'success': True, 'message': 'History cleared'})


@app.route('/export_history')
def export_history():
    """Export history as CSV"""
    df = export_history_to_csv()

    if df is None:
        return jsonify({'error': 'No history to export'}), 400

    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='prediction_history.csv'
    )


@app.route('/save_single', methods=['POST'])
def save_single_prediction():
    """Save a single prediction as JSON"""
    try:
        data = request.json

        # Create a formatted JSON string
        json_str = json.dumps(data, indent=2)

        output = BytesIO()
        output.write(json_str.encode('utf-8'))
        output.seek(0)

        return send_file(
            output,
            mimetype='application/json',
            as_attachment=True,
            download_name='exoplanet_prediction.json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/learn')
def learn():
    """Educational content about exoplanets"""
    facts = get_exoplanet_facts()
    feature_explanations = {name: get_feature_explanation(
        name) for name in feature_names}

    return jsonify({
        'facts': facts,
        'feature_explanations': feature_explanations
    })


@app.route('/model_info')
def model_info():
    """Get information about all models"""
    return jsonify(model_results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
