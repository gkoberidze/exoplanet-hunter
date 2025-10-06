import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_data():
    """Load the processed data"""
    print("Loading processed data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    feature_names = joblib.load('models/feature_names.pkl')

    return X_train, X_test, y_train, y_test, feature_names


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    return model


def train_neural_network(X_train, y_train):
    """Train Neural Network model"""
    print("\n" + "="*50)
    print("Training Neural Network...")
    print("="*50)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics"""

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)

    print(f"\n{model_name} Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test,
                                target_names=['False Positive', 'Confirmed Exoplanet']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)

    return {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }


def plot_model_comparison(results):
    """Create comparison plots for all models"""

    models = list(results.keys())
    test_accs = [results[m]['test_accuracy'] for m in models]
    roc_aucs = [results[m]['roc_auc'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    axes[0].bar(models, test_accs, color=['#667eea', '#764ba2', '#f093fb'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim([0.8, 1.0])
    for i, v in enumerate(test_accs):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

    # ROC AUC comparison
    axes[1].bar(models, roc_aucs, color=['#667eea', '#764ba2', '#f093fb'])
    axes[1].set_ylabel('ROC AUC Score')
    axes[1].set_title('Model ROC AUC Comparison')
    axes[1].set_ylim([0.8, 1.0])
    for i, v in enumerate(roc_aucs):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('static/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison saved to static/model_comparison.png")


def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['False Positive', 'Confirmed'],
                    yticklabels=['False Positive', 'Confirmed'])
        axes[idx].set_title(
            f'{model_name}\nAccuracy: {metrics["test_accuracy"]:.3f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('static/confusion_matrices_all.png',
                dpi=300, bbox_inches='tight')
    print("All confusion matrices saved to static/confusion_matrices_all.png")


def main():
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Train all models
    models = {
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train),
        'Neural Network': train_neural_network(X_train, y_train)
    }

    # Evaluate all models
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(
            model, X_train, X_test, y_train, y_test, name)

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = models[best_model_name]

    print("\n" + "="*50)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print("="*50)

    # Save all models
    for name, model in models.items():
        filename = name.lower().replace(' ', '_')
        joblib.dump(model, f'models/{filename}_model.pkl')
        print(f"Saved {name} to models/{filename}_model.pkl")

    # Save the best model as the default
    joblib.dump(best_model, 'models/exoplanet_classifier.pkl')
    print(f"\nBest model saved as default: models/exoplanet_classifier.pkl")

    # Save results
    with open('models/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to models/model_comparison.json")

    # Create visualizations
    plot_model_comparison(results)
    plot_confusion_matrices(results)

    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
                   rotation=45, ha='right')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('static/feature_importance_best.png',
                    dpi=300, bbox_inches='tight')
        print("Feature importance saved to static/feature_importance_best.png")

    print("\n✨ All models trained successfully! ✨")


if __name__ == "__main__":
    main()
