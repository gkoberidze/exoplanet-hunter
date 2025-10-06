import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_enhanced_data():
    """Load the enhanced processed data"""
    print("Loading enhanced data...")
    X_train = np.load('data/X_train_enhanced.npy')
    X_test = np.load('data/X_test_enhanced.npy')
    y_train = np.load('data/y_train_enhanced.npy')
    y_test = np.load('data/y_test_enhanced.npy')
    feature_names = joblib.load('models/feature_names_enhanced.pkl')

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {len(feature_names)}")

    return X_train, X_test, y_train, y_test, feature_names


def train_all_models(X_train, y_train):
    """Train all three models"""

    models = {}

    # Random Forest
    print("\n" + "="*60)
    print("🌲 Training Enhanced Random Forest...")
    print("="*60)
    models['Random Forest Enhanced'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    models['Random Forest Enhanced'].fit(X_train, y_train)

    # XGBoost
    print("\n" + "="*60)
    print("⚡ Training Enhanced XGBoost...")
    print("="*60)
    models['XGBoost Enhanced'] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    models['XGBoost Enhanced'].fit(X_train, y_train)

    # Neural Network
    print("\n" + "="*60)
    print("🧠 Training Enhanced Neural Network...")
    print("="*60)
    models['Neural Network Enhanced'] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001
    )
    models['Neural Network Enhanced'].fit(X_train, y_train)

    return models


def evaluate_all_models(models, X_train, X_test, y_train, y_test):
    """Evaluate all models"""

    results = {}

    for name, model in models.items():
        print("\n" + "="*60)
        print(f"📊 Evaluating {name}")
        print("="*60)

        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_proba_test)

        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test,
                                    target_names=['False Positive', 'Confirmed Exoplanet']))

        results[name] = {
            'test_accuracy': float(test_acc),
            'roc_auc': float(roc_auc)
        }

    return results


def create_final_comparison(results, feature_names):
    """Create comprehensive comparison visualization"""

    # Model comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = list(results.keys())
    test_accs = [results[m]['test_accuracy'] for m in models]
    roc_aucs = [results[m]['roc_auc'] for m in models]

    # Accuracy comparison
    axes[0, 0].bar(range(len(models)), test_accs, color=[
                   '#667eea', '#764ba2', '#f093fb'])
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Enhanced Models - Test Accuracy',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0.85, 0.95])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_accs):
        axes[0, 0].text(i, v + 0.002, f'{v:.4f}',
                        ha='center', fontweight='bold')

    # ROC AUC comparison
    axes[0, 1].bar(range(len(models)), roc_aucs, color=[
                   '#667eea', '#764ba2', '#f093fb'])
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
    axes[0, 1].set_ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Enhanced Models - ROC AUC',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0.85, 0.98])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(roc_aucs):
        axes[0, 1].text(i, v + 0.002, f'{v:.4f}',
                        ha='center', fontweight='bold')

    # Feature count info
    axes[1, 0].text(0.5, 0.7, f'Total Features Used',
                    ha='center', fontsize=16, fontweight='bold')
    axes[1, 0].text(0.5, 0.5, f'{len(feature_names)}',
                    ha='center', fontsize=48, fontweight='bold', color='#667eea')
    axes[1, 0].text(0.5, 0.3, 'Including derived features',
                    ha='center', fontsize=12, style='italic')
    axes[1, 0].axis('off')

    # Best model highlight
    best_model = max(results, key=lambda x: results[x]['test_accuracy'])
    best_acc = results[best_model]['test_accuracy']
    best_auc = results[best_model]['roc_auc']

    axes[1, 1].text(0.5, 0.75, '🏆 BEST MODEL',
                    ha='center', fontsize=18, fontweight='bold', color='gold')
    axes[1, 1].text(0.5, 0.6, best_model,
                    ha='center', fontsize=16, fontweight='bold', color='#667eea')
    axes[1, 1].text(0.5, 0.45, f'Accuracy: {best_acc:.4f}',
                    ha='center', fontsize=14)
    axes[1, 1].text(0.5, 0.35, f'ROC AUC: {best_auc:.4f}',
                    ha='center', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('static/enhanced_model_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\n📊 Enhanced model comparison saved to static/enhanced_model_comparison.png")


def main():
    # Load enhanced data
    X_train, X_test, y_train, y_test, feature_names = load_enhanced_data()

    # Train all models
    models = train_all_models(X_train, y_train)

    # Evaluate all models
    results = evaluate_all_models(models, X_train, X_test, y_train, y_test)

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = models[best_model_name]

    print("\n" + "="*60)
    print(f"🏆 BEST ENHANCED MODEL: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print("="*60)

    # Save all enhanced models
    for name, model in models.items():
        filename = name.lower().replace(' ', '_')
        joblib.dump(model, f'models/{filename}.pkl')
        print(f"💾 Saved {name}")

    # Save best as default
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(feature_names, 'models/best_feature_names.pkl')
    print(f"\n✨ Best model saved as: models/best_model.pkl")

    # Save results
    with open('models/enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Create visualizations
    create_final_comparison(results, feature_names)

    print("\n🎉 ALL ENHANCED MODELS TRAINED SUCCESSFULLY! 🎉")
    print(f"✅ {len(models)} models trained")
    print(f"✅ Using {len(feature_names)} features")
    print(f"✅ Best accuracy: {results[best_model_name]['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
