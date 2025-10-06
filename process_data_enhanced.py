import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_clean_data_enhanced(filepath):
    """Load and clean the exoplanet data with MORE features"""
    print("Loading data with enhanced features...")

    # Read the CSV
    try:
        df = pd.read_csv(filepath, comment='#')
    except:
        try:
            df = pd.read_csv(filepath, skiprows=50, comment='#')
        except:
            df = pd.read_csv(filepath, on_bad_lines='skip')

    print(f"Initial data shape: {df.shape}")

    # Use MORE features this time!
    possible_features = [
        # Original features
        'koi_period',      # Orbital period
        'koi_depth',       # Transit depth
        'koi_duration',    # Transit duration
        'koi_prad',        # Planetary radius
        'koi_teq',         # Equilibrium temperature
        'koi_steff',       # Stellar effective temperature
        'koi_srad',        # Stellar radius
        'koi_slogg',       # Stellar surface gravity

        # NEW features we're adding!
        'koi_insol',       # Insolation flux
        'koi_model_snr',   # Transit signal-to-noise ratio
        'koi_impact',      # Impact parameter
        'koi_kepmag',      # Kepler magnitude
        'koi_score',       # Disposition score
    ]

    # Only use features that exist
    features = [f for f in possible_features if f in df.columns]

    print(f"\nUsing {len(features)} features: {features}")

    # Target variable
    target = 'koi_disposition'

    # Keep only rows with these columns
    df_clean = df[features + [target]].copy()

    # Remove rows with missing values
    df_clean = df_clean.dropna()

    # Create DERIVED features (feature engineering!)
    if 'koi_prad' in features and 'koi_srad' in features:
        df_clean['planet_star_radius_ratio'] = df_clean['koi_prad'] / \
            df_clean['koi_srad']
        features.append('planet_star_radius_ratio')
        print("✨ Added derived feature: planet_star_radius_ratio")

    if 'koi_period' in features and 'koi_duration' in features:
        df_clean['duration_period_ratio'] = df_clean['koi_duration'] / \
            df_clean['koi_period']
        features.append('duration_period_ratio')
        print("✨ Added derived feature: duration_period_ratio")

    # Convert target to binary
    df_clean = df_clean[df_clean[target].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    df_clean[target] = (df_clean[target] == 'CONFIRMED').astype(int)

    print(f"\nFinal data loaded: {len(df_clean)} samples")
    print(f"Confirmed exoplanets: {df_clean[target].sum()}")
    print(f"False positives: {len(df_clean) - df_clean[target].sum()}")
    print(f"Total features: {len(features)}")

    return df_clean, features, target


def prepare_data(df, features, target):
    """Split and scale the data"""
    X = df[features]
    y = df[target]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, 'models/scaler_enhanced.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Process the data with enhanced features
    df, features, target = load_and_clean_data_enhanced('data/kepler_data.csv')
    X_train, X_test, y_train, y_test = prepare_data(df, features, target)

    # Save processed data
    np.save('data/X_train_enhanced.npy', X_train)
    np.save('data/X_test_enhanced.npy', X_test)
    np.save('data/y_train_enhanced.npy', y_train)
    np.save('data/y_test_enhanced.npy', y_test)

    # Save feature names
    joblib.dump(features, 'models/feature_names_enhanced.pkl')

    print("\n✨ Enhanced data processing complete! ✨")
    print("Files saved with '_enhanced' suffix")
