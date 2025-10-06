import json
import os
from datetime import datetime
import pandas as pd

HISTORY_FILE = 'data/prediction_history.json'


def save_prediction_to_history(prediction_data):
    """Save a prediction to history file"""

    # Add timestamp
    prediction_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Load existing history
    history = load_prediction_history()

    # Add new prediction
    history.append(prediction_data)

    # Keep only last 100 predictions
    if len(history) > 100:
        history = history[-100:]

    # Save back to file
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    return True


def load_prediction_history():
    """Load prediction history from file"""
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []


def clear_prediction_history():
    """Clear all prediction history"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return True


def export_history_to_csv():
    """Export prediction history to CSV format"""
    history = load_prediction_history()

    if not history:
        return None

    # Flatten the data for CSV
    flat_data = []
    for item in history:
        flat_item = {
            'timestamp': item.get('timestamp', ''),
            'prediction': item.get('prediction', ''),
            'confidence': item.get('confidence', ''),
            'model_used': item.get('model_used', '')
        }

        # Add input values
        if 'input_values' in item:
            for key, value in item['input_values'].items():
                flat_item[f'input_{key}'] = value

        flat_data.append(flat_item)

    return pd.DataFrame(flat_data)
