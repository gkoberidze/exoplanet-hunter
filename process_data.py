import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_clean_data(filepath):
    """Load and clean the exoplanet data"""
    print("Loading data...")

    # Try to read with different settings to handle NASA format
    try:
        df = pd.read_csv(filepath, comment='#')
    except:
        try:
            df = pd.read_csv(filepath, skiprows=50, comment='#')
        except:
            df = pd.read_csv(filepath, on_bad_lines='skip')

    print(f"Initial data shape: {df.shape}")

    # Select relevant features that exist in the dataset
    possible_features = [
        'koi_period',      # Orbital period
        'koi_depth',       # Transit depth
        'koi_duration',    # Transit duration
        'koi_prad',        # Planetary radius
        'koi_teq',         # Equilibrium temperature
        'koi_steff',       # Stellar effective temperature
        'koi_srad',        # Stellar radius
        'koi_slogg',       # Stellar surface gravity
    ]

    # Only use features that actually exist in the dataframe
    features = [f for f in possible_features if f in df.columns]

    print(f"Using features: {features}")

    # Target variable (what we're predicting)
    target = 'koi_disposition'

    # Keep only rows with these columns
    df_clean = df[features + [target]].copy()

    # Remove rows with missing values
    df_clean = df_clean.dropna()

    # Convert target to binary: CONFIRMED = 1, FALSE POSITIVE = 0, remove CANDIDATE
    df_clean = df_clean[df_clean[target].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    df_clean[target] = (df_clean[target] == 'CONFIRMED').astype(int)

    print(f"\nData loaded: {len(df_clean)} samples")
    print(f"Confirmed exoplanets: {df_clean[target].sum()}")
    print(f"False positives: {len(df_clean) - df_clean[target].sum()}")

    return df_clean, features, target


def prepare_data(df, features, target):
    """Split and scale the data"""
    X = df[features]
    y = df[target]

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features (important for many ML algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Process the data
    df, features, target = load_and_clean_data('data/kepler_data.csv')
    X_train, X_test, y_train, y_test = prepare_data(df, features, target)

    # Save processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

    # Save feature names
    joblib.dump(features, 'models/feature_names.pkl')

    print("\nData processing complete!")
    print("Files saved in data/ and models/ folders")
