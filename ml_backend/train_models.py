"""
AI Path Logistics Agent — ML Model Training Pipeline
=====================================================
Trains 3 models on the Dynamic Supply Chain Logistics Dataset (32K rows):

1. Delay Probability Regressor (Random Forest)
2. Risk Classification Model (Random Forest Classifier)
3. Fuel Optimization Regressor (Gradient Boosting)

All models are saved as .joblib files in ml_backend/models/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, f1_score
)

warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dynamic_supply_chain_logistics_dataset.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature sets for each model
DELAY_FEATURES = [
    'traffic_congestion_level', 'weather_condition_severity',
    'fuel_consumption_rate', 'route_risk_level',
    'driver_behavior_score', 'fatigue_monitoring_score',
    'loading_unloading_time', 'eta_variation_hours',
    'port_congestion_level', 'supplier_reliability_score',
    'disruption_likelihood_score'
]

RISK_FEATURES = [
    'traffic_congestion_level', 'weather_condition_severity',
    'fuel_consumption_rate', 'route_risk_level',
    'driver_behavior_score', 'fatigue_monitoring_score',
    'loading_unloading_time', 'eta_variation_hours',
    'port_congestion_level', 'supplier_reliability_score',
    'disruption_likelihood_score', 'delay_probability',
    'shipping_costs', 'customs_clearance_time'
]

FUEL_FEATURES = [
    'traffic_congestion_level', 'weather_condition_severity',
    'route_risk_level', 'driver_behavior_score',
    'delivery_time_deviation', 'loading_unloading_time',
    'eta_variation_hours', 'fatigue_monitoring_score'
]


def load_data():
    """Load and preprocess the supply chain dataset."""
    print("[*] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df):,} rows x {len(df.columns)} columns")
    
    # Collect all required columns (deduplicated)
    all_features = list(set(DELAY_FEATURES + RISK_FEATURES + FUEL_FEATURES))
    targets = ['delay_probability', 'risk_classification', 'fuel_consumption_rate', 'delivery_time_deviation']
    relevant_cols = list(dict.fromkeys(all_features + targets))  # preserve order, deduplicate
    df_clean = df[relevant_cols].dropna()
    print(f"   After cleaning: {len(df_clean):,} rows")
    
    return df_clean


def train_delay_model(df):
    """Train the Delay Probability Regressor."""
    print("\n[MODEL 1] Training Delay Probability Regressor")
    print("   Algorithm: Random Forest (200 trees, max_depth=12)")
    
    X = df[DELAY_FEATURES].values
    y = df['delay_probability'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   [OK] MAE: {mae:.4f}")
    print(f"   [OK] RMSE: {rmse:.4f}")
    print(f"   [OK] R2 Score: {r2:.4f}")
    
    # Feature importance
    importances = dict(zip(DELAY_FEATURES, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   [FEATURES] Top: {', '.join(f'{k}({v:.3f})' for k, v in top_features)}")
    
    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, 'delay_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'delay_scaler.joblib'))
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'features': DELAY_FEATURES,
        'top_features': {k: round(v, 4) for k, v in top_features}
    }


def train_risk_model(df):
    """Train the Risk Classification Model."""
    print("\n[MODEL 2] Training Risk Classification Model")
    print("   Algorithm: Random Forest Classifier (200 trees, max_depth=15)")
    
    X = df[RISK_FEATURES].values
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['risk_classification'].values)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with class weights to handle imbalance
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    class_names = le.classes_.tolist()
    print(f"   [OK] Accuracy: {accuracy:.4f}")
    print(f"   [OK] F1 Score (weighted): {f1:.4f}")
    print(f"   [CLASSES] {class_names}")
    
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    for cls in class_names:
        p = report[cls]['precision']
        r = report[cls]['recall']
        print(f"      {cls}: precision={p:.3f}, recall={r:.3f}")
    
    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, 'risk_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'risk_scaler.joblib'))
    joblib.dump(le, os.path.join(MODELS_DIR, 'risk_label_encoder.joblib'))
    
    return {
        'accuracy': round(accuracy, 4),
        'f1_weighted': round(f1, 4),
        'classes': class_names,
        'features': RISK_FEATURES
    }


def train_fuel_model(df):
    """Train the Fuel Optimization Regressor."""
    print("\n[MODEL 3] Training Fuel Optimization Regressor")
    print("   Algorithm: Gradient Boosting (300 trees, max_depth=6)")
    
    X = df[FUEL_FEATURES].values
    y = df['fuel_consumption_rate'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   [OK] MAE: {mae:.4f} L/100km")
    print(f"   [OK] RMSE: {rmse:.4f}")
    print(f"   [OK] R2 Score: {r2:.4f}")
    
    # Feature importance
    importances = dict(zip(FUEL_FEATURES, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   [FEATURES] Top: {', '.join(f'{k}({v:.3f})' for k, v in top_features)}")
    
    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, 'fuel_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'fuel_scaler.joblib'))
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'features': FUEL_FEATURES,
        'top_features': {k: round(v, 4) for k, v in top_features}
    }


def main():
    print("=" * 70)
    print("  AI PATH LOGISTICS - ML MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    df = load_data()
    
    # Train all 3 models
    delay_metrics = train_delay_model(df)
    risk_metrics = train_risk_model(df)
    fuel_metrics = train_fuel_model(df)
    
    # Save metadata
    metadata = {
        'dataset_rows': len(df),
        'models': {
            'delay_probability': {
                'file': 'delay_model.joblib',
                'scaler': 'delay_scaler.joblib',
                'algorithm': 'RandomForestRegressor',
                'metrics': delay_metrics
            },
            'risk_classification': {
                'file': 'risk_model.joblib',
                'scaler': 'risk_scaler.joblib',
                'label_encoder': 'risk_label_encoder.joblib',
                'algorithm': 'RandomForestClassifier',
                'metrics': risk_metrics
            },
            'fuel_optimization': {
                'file': 'fuel_model.joblib',
                'scaler': 'fuel_scaler.joblib',
                'algorithm': 'GradientBoostingRegressor',
                'metrics': fuel_metrics
            }
        }
    }
    
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  [DONE] ALL MODELS TRAINED SUCCESSFULLY")
    print(f"  [DIR] Models saved to: {os.path.abspath(MODELS_DIR)}")
    print(f"  [META] Metadata saved to: {os.path.abspath(meta_path)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
