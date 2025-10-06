import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_uploaded_csv(file_path, scaler, model, feature_names):
    """Process an uploaded CSV file and make predictions"""

    try:
        # Read the CSV
        df = pd.read_csv(file_path)

        print(f"Uploaded CSV columns: {df.columns.tolist()}")
        print(f"Required features: {feature_names}")

        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            return {
                'error': f'Missing required columns: {", ".join(missing_features)}',
                'required_columns': feature_names
            }

        # Extract features
        X = df[feature_names].copy()

        # Store original indices
        original_indices = X.index.tolist()

        # Handle missing values
        rows_with_data = X.dropna()
        dropped_count = len(X) - len(rows_with_data)

        if len(rows_with_data) == 0:
            return {'error': 'No valid rows found in the uploaded file (all rows have missing values)'}

        # Scale the features
        X_scaled = scaler.transform(rows_with_data)

        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Create results dataframe
        results_df = rows_with_data.copy()
        results_df['prediction'] = ['Confirmed Exoplanet' if p ==
                                    1 else 'False Positive' for p in predictions]
        results_df['confidence'] = [max(prob) * 100 for prob in probabilities]
        results_df['probability_exoplanet'] = [
            prob[1] * 100 for prob in probabilities]
        results_df['probability_false_positive'] = [
            prob[0] * 100 for prob in probabilities]

        # Summary statistics
        total_analyzed = len(predictions)
        confirmed_count = sum(predictions)
        false_positive_count = total_analyzed - confirmed_count
        avg_confidence = np.mean([max(prob) for prob in probabilities]) * 100

        summary = {
            'total_rows': len(df),
            'analyzed_rows': total_analyzed,
            'dropped_rows': dropped_count,
            'confirmed_exoplanets': int(confirmed_count),
            'false_positives': int(false_positive_count),
            'average_confidence': float(avg_confidence)
        }

        # Convert results to list of dicts for JSON
        results_list = results_df.to_dict('records')

        return {
            'success': True,
            'summary': summary,
            'results': results_list,
            'columns': list(results_df.columns)
        }

    except Exception as e:
        return {'error': f'Error processing file: {str(e)}'}
