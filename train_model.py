import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def train_model():
    """Train a Random Forest model"""
    print("Loading processed data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    print("Training model...")
    # Random Forest is a good starting model
    model = RandomForestClassifier(
        n_estimators=200,      # Number of trees
        max_depth=20,          # Maximum depth of trees
        min_samples_split=5,   # Minimum samples to split a node
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test,
                                target_names=['False Positive', 'Confirmed Exoplanet']))

    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('static/confusion_matrix.png')
    print("\nConfusion matrix saved to static/confusion_matrix.png")

    # Save the model
    joblib.dump(model, 'models/exoplanet_classifier.pkl')
    print("Model saved to models/exoplanet_classifier.pkl")

    # Feature importance
    feature_names = joblib.load('models/feature_names.pkl')
    importances = model.feature_importances_

    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i]
               for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    print("Feature importance saved to static/feature_importance.png")

    return model


if __name__ == "__main__":
    model = train_model()
    print("\nTraining complete! ✨")
