"""
AI Path Logistics Agent — ML Prediction API
=============================================
Flask API serving 3 trained ML models for logistics optimization:
- /api/predict/delay    — Delay probability prediction
- /api/predict/risk     — Risk classification
- /api/predict/fuel     — Fuel optimization recommendation
- /api/predict/decision — Combined AI decision generator
- /api/health           — Health check + model metrics
"""

import os
import json
import random
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ───────────────────────────────────────────────────────────
# Load Models
# ───────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model_assets():
    """Load all trained models, scalers, and metadata."""
    assets = {}
    try:
        assets['delay_model'] = joblib.load(os.path.join(MODELS_DIR, 'delay_model.joblib'))
        assets['delay_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'delay_scaler.joblib'))
        
        assets['risk_model'] = joblib.load(os.path.join(MODELS_DIR, 'risk_model.joblib'))
        assets['risk_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'risk_scaler.joblib'))
        assets['risk_le'] = joblib.load(os.path.join(MODELS_DIR, 'risk_label_encoder.joblib'))
        
        assets['fuel_model'] = joblib.load(os.path.join(MODELS_DIR, 'fuel_model.joblib'))
        assets['fuel_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'fuel_scaler.joblib'))
        
        meta_path = os.path.join(MODELS_DIR, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                assets['metadata'] = json.load(f)
        
        assets['loaded'] = True
        print("[OK] All models loaded successfully")
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")
        assets['loaded'] = False
    
    return assets

models = load_model_assets()

# ───────────────────────────────────────────────────────────
# Feature Definitions (must match training)
# ───────────────────────────────────────────────────────────

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

# ───────────────────────────────────────────────────────────
# Helper: Map vehicle data to model features
# ───────────────────────────────────────────────────────────

def vehicle_to_features(vehicle_data):
    """
    Convert vehicle/operational data from the frontend into 
    feature vectors for each model. Uses sensible defaults for
    features that don't come from the frontend directly.
    """
    fuel_level = vehicle_data.get('fuel_level', 50)
    status = vehicle_data.get('status', 'idle')
    load_pct = vehicle_data.get('load_percentage', 50)
    
    # Derive contextual features from vehicle state
    # These approximate the dataset's feature distributions
    traffic = vehicle_data.get('traffic_congestion_level', random.uniform(1, 8))
    weather = vehicle_data.get('weather_condition_severity', random.uniform(0, 5))
    fuel_rate = vehicle_data.get('fuel_consumption_rate', 5 + (100 - fuel_level) * 0.1)
    route_risk = vehicle_data.get('route_risk_level', random.uniform(1, 8))
    driver_score = vehicle_data.get('driver_behavior_score', random.uniform(0.3, 0.95))
    fatigue = vehicle_data.get('fatigue_monitoring_score', random.uniform(0.2, 0.95))
    loading_time = vehicle_data.get('loading_unloading_time', random.uniform(1, 5))
    eta_var = vehicle_data.get('eta_variation_hours', random.uniform(0, 5))
    port_congestion = vehicle_data.get('port_congestion_level', random.uniform(1, 8))
    supplier_reliability = vehicle_data.get('supplier_reliability_score', random.uniform(0.3, 0.99))
    disruption = vehicle_data.get('disruption_likelihood_score', random.uniform(0.1, 0.9))
    shipping_costs = vehicle_data.get('shipping_costs', random.uniform(100, 800))
    customs_time = vehicle_data.get('customs_clearance_time', random.uniform(0.1, 1.0))
    delivery_deviation = vehicle_data.get('delivery_time_deviation', random.uniform(0, 8))
    
    # Adjust features based on vehicle status
    if status == 'moving':
        traffic *= 1.2
        fatigue *= 1.1
    elif status == 'delayed':
        traffic *= 1.8
        disruption *= 1.3
        eta_var *= 1.5
    elif status == 'maintenance':
        route_risk *= 0.5
    
    # Low fuel increases risk
    if fuel_level < 25:
        disruption *= 1.4
        route_risk *= 1.3
    
    # High load increases fuel consumption
    fuel_rate += load_pct * 0.05
    
    return {
        'traffic_congestion_level': min(traffic, 10),
        'weather_condition_severity': min(weather, 10),
        'fuel_consumption_rate': min(fuel_rate, 20),
        'route_risk_level': min(route_risk, 10),
        'driver_behavior_score': min(max(driver_score, 0), 1),
        'fatigue_monitoring_score': min(max(fatigue, 0), 1),
        'loading_unloading_time': max(loading_time, 0),
        'eta_variation_hours': max(eta_var, 0),
        'port_congestion_level': min(port_congestion, 10),
        'supplier_reliability_score': min(max(supplier_reliability, 0), 1),
        'disruption_likelihood_score': min(max(disruption, 0), 1),
        'shipping_costs': max(shipping_costs, 0),
        'customs_clearance_time': max(customs_time, 0),
        'delivery_time_deviation': delivery_deviation,
        'delay_probability': 0.5,  # Will be filled by model 1
    }


def make_feature_vector(features_dict, feature_names):
    """Extract ordered feature vector from dict."""
    return np.array([[features_dict[f] for f in feature_names]])


# ───────────────────────────────────────────────────────────
# ENDPOINTS
# ───────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with model metrics."""
    return jsonify({
        'status': 'healthy' if models.get('loaded') else 'models_not_loaded',
        'models_loaded': models.get('loaded', False),
        'timestamp': datetime.now().isoformat(),
        'metadata': models.get('metadata', {}),
        'endpoints': [
            '/api/predict/delay',
            '/api/predict/risk',
            '/api/predict/fuel',
            '/api/predict/decision',
        ]
    })


@app.route('/api/predict/delay', methods=['POST'])
def predict_delay():
    """Predict delay probability for given conditions."""
    if not models.get('loaded'):
        return jsonify({'error': 'Models not loaded'}), 503
    
    data = request.json
    features = vehicle_to_features(data)
    X = make_feature_vector(features, DELAY_FEATURES)
    X_scaled = models['delay_scaler'].transform(X)
    
    delay_prob = float(models['delay_model'].predict(X_scaled)[0])
    delay_prob = max(0.0, min(1.0, delay_prob))
    
    # Estimate time deviation based on delay probability
    time_deviation_hours = delay_prob * 8.0  # max ~8 hours deviation
    
    return jsonify({
        'delay_probability': round(delay_prob, 4),
        'delay_percentage': round(delay_prob * 100, 1),
        'estimated_deviation_hours': round(time_deviation_hours, 2),
        'risk_level': 'critical' if delay_prob > 0.8 else 'high' if delay_prob > 0.6 else 'medium' if delay_prob > 0.3 else 'low',
        'features_used': {k: round(features[k], 3) for k in DELAY_FEATURES}
    })


@app.route('/api/predict/risk', methods=['POST'])
def predict_risk():
    """Classify shipment risk level."""
    if not models.get('loaded'):
        return jsonify({'error': 'Models not loaded'}), 503
    
    data = request.json
    features = vehicle_to_features(data)
    X = make_feature_vector(features, RISK_FEATURES)
    X_scaled = models['risk_scaler'].transform(X)
    
    risk_class_idx = models['risk_model'].predict(X_scaled)[0]
    risk_label = models['risk_le'].inverse_transform([risk_class_idx])[0]
    
    # Get probability distribution
    probas = models['risk_model'].predict_proba(X_scaled)[0]
    class_names = models['risk_le'].classes_.tolist()
    confidence = {name: round(float(prob), 4) for name, prob in zip(class_names, probas)}
    
    return jsonify({
        'risk_classification': risk_label,
        'confidence': confidence,
        'primary_confidence': round(float(max(probas)), 4),
        'features_used': {k: round(features[k], 3) for k in RISK_FEATURES}
    })


@app.route('/api/predict/fuel', methods=['POST'])
def predict_fuel():
    """Predict optimal fuel consumption rate."""
    if not models.get('loaded'):
        return jsonify({'error': 'Models not loaded'}), 503
    
    data = request.json
    features = vehicle_to_features(data)
    X = make_feature_vector(features, FUEL_FEATURES)
    X_scaled = models['fuel_scaler'].transform(X)
    
    optimal_fuel = float(models['fuel_model'].predict(X_scaled)[0])
    actual_fuel = features['fuel_consumption_rate']
    
    savings_pct = max(0, ((actual_fuel - optimal_fuel) / actual_fuel) * 100)
    # At ₹95/L fuel cost, estimate savings per 100km
    savings_inr_per_100km = max(0, (actual_fuel - optimal_fuel) * 95)
    
    return jsonify({
        'optimal_fuel_rate': round(optimal_fuel, 3),
        'current_fuel_rate': round(actual_fuel, 3),
        'potential_savings_percent': round(savings_pct, 1),
        'estimated_savings_inr_per_100km': round(savings_inr_per_100km, 0),
        'fuel_cost_per_liter_inr': 95,
        'features_used': {k: round(features[k], 3) for k in FUEL_FEATURES}
    })


@app.route('/api/predict/decision', methods=['POST'])
def predict_decision():
    """
    Master endpoint: Generate AI decisions for a list of vehicles.
    Runs all 3 models and produces structured AIDecision objects 
    matching the frontend's expected format.
    """
    if not models.get('loaded'):
        return jsonify({'error': 'Models not loaded'}), 503
    
    data = request.json
    vehicles = data.get('vehicles', [])
    
    if not vehicles:
        return jsonify({'error': 'No vehicles provided'}), 400
    
    decisions = []
    
    for vehicle in vehicles:
        features = vehicle_to_features(vehicle)
        
        # ── Model 1: Delay Prediction ──
        X_delay = make_feature_vector(features, DELAY_FEATURES)
        X_delay_scaled = models['delay_scaler'].transform(X_delay)
        delay_prob = float(models['delay_model'].predict(X_delay_scaled)[0])
        delay_prob = max(0.0, min(1.0, delay_prob))
        
        # Feed delay_prob into risk model features
        features['delay_probability'] = delay_prob
        
        # ── Model 2: Risk Classification ──
        X_risk = make_feature_vector(features, RISK_FEATURES)
        X_risk_scaled = models['risk_scaler'].transform(X_risk)
        risk_idx = models['risk_model'].predict(X_risk_scaled)[0]
        risk_label = models['risk_le'].inverse_transform([risk_idx])[0]
        risk_probas = models['risk_model'].predict_proba(X_risk_scaled)[0]
        risk_confidence = round(float(max(risk_probas)) * 100, 1)
        
        # ── Model 3: Fuel Optimization ──
        X_fuel = make_feature_vector(features, FUEL_FEATURES)
        X_fuel_scaled = models['fuel_scaler'].transform(X_fuel)
        optimal_fuel = float(models['fuel_model'].predict(X_fuel_scaled)[0])
        actual_fuel = features['fuel_consumption_rate']
        fuel_gap_pct = max(0, ((actual_fuel - optimal_fuel) / actual_fuel) * 100) if actual_fuel > 0 else 0
        
        # ── Decision Logic ──
        plate = vehicle.get('plate_number', 'Unknown')
        city = vehicle.get('city', 'Unknown')
        vehicle_id = vehicle.get('id', '')
        
        decision_type, severity, reasoning, action, profit_delta, time_delta = generate_decision(
            delay_prob=delay_prob,
            risk_label=risk_label,
            risk_confidence=risk_confidence,
            fuel_gap_pct=fuel_gap_pct,
            optimal_fuel=optimal_fuel,
            actual_fuel=actual_fuel,
            plate=plate,
            city=city,
            features=features
        )
        
        decisions.append({
            'vehicle_id': vehicle_id,
            'decision_type': decision_type,
            'severity': severity,
            'reasoning': reasoning,
            'proposed_action': action,
            'profit_delta_inr': profit_delta,
            'time_delta_minutes': time_delta,
            'utilization_delta': round(random.uniform(-0.1, 0.15), 3),
            'ml_metadata': {
                'delay_probability': round(delay_prob, 4),
                'risk_classification': risk_label,
                'risk_confidence': risk_confidence,
                'optimal_fuel_rate': round(optimal_fuel, 2),
                'fuel_efficiency_gap': round(fuel_gap_pct, 1),
            }
        })
    
    return jsonify({
        'success': True,
        'decisions_generated': len(decisions),
        'decisions': decisions,
        'model_version': 'ml-v1.0',
        'timestamp': datetime.now().isoformat()
    })


def generate_decision(delay_prob, risk_label, risk_confidence, fuel_gap_pct, 
                       optimal_fuel, actual_fuel, plate, city, features):
    """
    Core decision logic combining all 3 model outputs into an actionable recommendation.
    Returns: (decision_type, severity, reasoning, proposed_action, profit_delta_inr, time_delta_minutes)
    """
    traffic = features['traffic_congestion_level']
    weather = features['weather_condition_severity']
    driver_score = features['driver_behavior_score']
    fatigue = features['fatigue_monitoring_score']
    
    # Priority 1: High delay risk → REROUTE
    if delay_prob > 0.7:
        severity = 'critical' if delay_prob > 0.85 else 'high'
        time_saved = int(delay_prob * 45)
        profit = int(time_saved * 350 + random.randint(500, 3000))
        
        if traffic > 6:
            reasoning = (f"ML model predicts {delay_prob*100:.1f}% delay probability for {plate} "
                        f"near {city}. Traffic congestion level at {traffic:.1f}/10 is significantly "
                        f"above the safe threshold. Weather severity: {weather:.1f}/10. "
                        f"Risk classification: {risk_label} ({risk_confidence}% confidence).")
            action = (f"Reroute via alternative corridor to bypass congestion. "
                     f"Estimated time savings: {time_saved} minutes. "
                     f"Projected savings: ₹{profit:,} in operational costs.")
        else:
            reasoning = (f"ML delay model flags {delay_prob*100:.1f}% delay risk for {plate}. "
                        f"Weather severity ({weather:.1f}/10) and disruption likelihood "
                        f"({features['disruption_likelihood_score']:.2f}) are key factors. "
                        f"Overall risk: {risk_label}.")
            action = (f"Preemptively adjust departure schedule and notify destination warehouse. "
                     f"Consider weather-safe rest stop near {city}. "
                     f"Estimated impact: save {time_saved} min, ₹{profit:,}.")
        
        return 'REROUTE', severity, reasoning, action, profit, -time_saved
    
    # Priority 2: Fuel inefficiency → SPEED_ADJUSTMENT
    if fuel_gap_pct > 15:
        severity = 'high' if fuel_gap_pct > 30 else 'medium'
        savings_per_trip = int(fuel_gap_pct * 180 + random.randint(200, 1500))
        
        reasoning = (f"Fuel optimization model detects {plate} consuming {actual_fuel:.1f} L/100km, "
                    f"while optimal rate is {optimal_fuel:.1f} L/100km ({fuel_gap_pct:.1f}% above optimal). "
                    f"Driver behavior score: {driver_score:.2f}/1.0, fatigue: {fatigue:.2f}/1.0. "
                    f"At ₹95/L, this represents significant cost overrun.")
        action = (f"Reduce speed to 60-65 km/h for optimal fuel efficiency. "
                 f"Schedule driver fatigue break if score < 0.5. "
                 f"Estimated savings: ₹{savings_per_trip:,} per trip at current fuel prices.")
        
        return 'SPEED_ADJUSTMENT', severity, reasoning, action, savings_per_trip, 15
    
    # Priority 3: High risk classification → PREVENTIVE_MAINTENANCE
    if risk_label == 'High Risk' and risk_confidence > 60:
        severity = 'high' if risk_confidence > 80 else 'medium'
        cost_avoidance = int(risk_confidence * 250 + random.randint(1000, 5000))
        
        reasoning = (f"Risk classifier assigns '{risk_label}' to {plate} with {risk_confidence}% "
                    f"confidence. Key risk factors: traffic ({traffic:.1f}), weather ({weather:.1f}), "
                    f"fatigue monitoring ({fatigue:.2f}). Preventive action recommended to avoid "
                    f"costly breakdown on highway near {city}.")
        action = (f"Schedule preventive inspection at nearest service center. "
                 f"Check braking system and tire pressure. "
                 f"Estimated breakdown cost avoidance: ₹{cost_avoidance:,}.")
        
        return 'PREVENTIVE_MAINTENANCE', severity, reasoning, action, cost_avoidance, 30
    
    # Priority 4: Moderate delay + available capacity → LOAD_MATCH
    if delay_prob > 0.3 and delay_prob <= 0.7:
        severity = 'medium' if delay_prob > 0.5 else 'low'
        revenue = int(random.randint(12000, 28000))
        
        reasoning = (f"ML models detect moderate delay risk ({delay_prob*100:.1f}%) for {plate}. "
                    f"Risk level: {risk_label}. Current conditions suggest opportunity for "
                    f"backhaul load matching to improve utilization and offset route costs. "
                    f"Supplier reliability in region: {features['supplier_reliability_score']:.2f}.")
        action = (f"Search for compatible backhaul loads from {city}. "
                 f"Accept loads with ≥80% route overlap to maximize revenue. "
                 f"Potential additional revenue: ₹{revenue:,}.")
        
        return 'LOAD_MATCH', severity, reasoning, action, revenue, 20
    
    # Default: Low risk → EMPTY_MILE_FIX
    severity = 'low'
    savings = int(random.randint(2000, 8000))
    
    reasoning = (f"All systems nominal for {plate} near {city}. Delay risk: {delay_prob*100:.1f}%, "
                f"Risk: {risk_label}, Fuel efficiency gap: {fuel_gap_pct:.1f}%. "
                f"ML models suggest optimizing empty miles on return route.")
    action = (f"Monitor for backhaul opportunities before return journey. "
             f"Current driver performance score: {driver_score:.2f}/1.0 — commendable. "
             f"Estimated empty-mile savings: ₹{savings:,}.")
    
    return 'EMPTY_MILE_FIX', severity, reasoning, action, savings, 0


# ───────────────────────────────────────────────────────────
# Load Accept/Reject Decision
# ───────────────────────────────────────────────────────────

# Approximate distances between major Indian cities (km)
CITY_DISTANCES = {
    ('Mumbai', 'Delhi'): 1400, ('Delhi', 'Mumbai'): 1400,
    ('Mumbai', 'Bangalore'): 980, ('Bangalore', 'Mumbai'): 980,
    ('Mumbai', 'Chennai'): 1340, ('Chennai', 'Mumbai'): 1340,
    ('Mumbai', 'Kolkata'): 2050, ('Kolkata', 'Mumbai'): 2050,
    ('Mumbai', 'Hyderabad'): 710, ('Hyderabad', 'Mumbai'): 710,
    ('Mumbai', 'Ahmedabad'): 524, ('Ahmedabad', 'Mumbai'): 524,
    ('Mumbai', 'Pune'): 150, ('Pune', 'Mumbai'): 150,
    ('Delhi', 'Kolkata'): 1530, ('Kolkata', 'Delhi'): 1530,
    ('Delhi', 'Bangalore'): 2150, ('Bangalore', 'Delhi'): 2150,
    ('Delhi', 'Chennai'): 2180, ('Chennai', 'Delhi'): 2180,
    ('Delhi', 'Hyderabad'): 1570, ('Hyderabad', 'Delhi'): 1570,
    ('Delhi', 'Jaipur'): 280, ('Jaipur', 'Delhi'): 280,
    ('Delhi', 'Lucknow'): 555, ('Lucknow', 'Delhi'): 555,
    ('Bangalore', 'Chennai'): 350, ('Chennai', 'Bangalore'): 350,
    ('Bangalore', 'Hyderabad'): 570, ('Hyderabad', 'Bangalore'): 570,
    ('Kolkata', 'Chennai'): 1660, ('Chennai', 'Kolkata'): 1660,
    ('Ahmedabad', 'Pune'): 660, ('Pune', 'Ahmedabad'): 660,
    ('Ahmedabad', 'Delhi'): 950, ('Delhi', 'Ahmedabad'): 950,
    ('Jaipur', 'Ahmedabad'): 670, ('Ahmedabad', 'Jaipur'): 670,
    ('Lucknow', 'Kolkata'): 990, ('Kolkata', 'Lucknow'): 990,
    ('Lucknow', 'Nagpur'): 860, ('Nagpur', 'Lucknow'): 860,
    ('Nagpur', 'Mumbai'): 840, ('Mumbai', 'Nagpur'): 840,
    ('Nagpur', 'Hyderabad'): 500, ('Hyderabad', 'Nagpur'): 500,
    ('Indore', 'Mumbai'): 585, ('Mumbai', 'Indore'): 585,
    ('Indore', 'Delhi'): 810, ('Delhi', 'Indore'): 810,
    ('Indore', 'Surat'): 390, ('Surat', 'Indore'): 390,
    ('Pune', 'Bangalore'): 840, ('Bangalore', 'Pune'): 840,
    ('Surat', 'Mumbai'): 285, ('Mumbai', 'Surat'): 285,
    ('Kanpur', 'Delhi'): 440, ('Delhi', 'Kanpur'): 440,
    ('Kanpur', 'Lucknow'): 80, ('Lucknow', 'Kanpur'): 80,
}

def get_distance(origin, destination):
    """Get approximate distance between two Indian cities."""
    key = (origin, destination)
    if key in CITY_DISTANCES:
        return CITY_DISTANCES[key]
    # Default estimate based on average intercity distance
    return 800


@app.route('/api/predict/load-decision', methods=['POST'])
def predict_load_decision():
    """
    Evaluate whether a vehicle should ACCEPT or REJECT a load.
    Considers: ML risk/delay predictions, route profitability, fuel cost,
    vehicle capacity, and detour distance.
    """
    if not models.get('loaded'):
        return jsonify({'error': 'Models not loaded'}), 503

    data = request.json
    vehicle = data.get('vehicle', {})
    load = data.get('load', {})

    if not vehicle or not load:
        return jsonify({'error': 'Both vehicle and load data required'}), 400

    # ── Extract load details ──
    origin = load.get('origin', 'Unknown')
    destination = load.get('destination', 'Unknown')
    weight_kg = load.get('weight_kg', 10000)
    price_inr = load.get('price_inr', 50000)
    vehicle_city = vehicle.get('city', 'Unknown')
    vehicle_capacity = vehicle.get('capacity_kg', 28000)
    current_load = vehicle.get('current_load_kg', 0)
    fuel_level = vehicle.get('fuel_level', 50)
    vehicle_status = vehicle.get('status', 'idle')
    plate = vehicle.get('plate_number', 'Unknown')

    # ── Run ML Models ──
    features = vehicle_to_features(vehicle)
    
    # Model 1: Delay
    X_delay = make_feature_vector(features, DELAY_FEATURES)
    X_delay_scaled = models['delay_scaler'].transform(X_delay)
    delay_prob = float(models['delay_model'].predict(X_delay_scaled)[0])
    delay_prob = max(0.0, min(1.0, delay_prob))
    features['delay_probability'] = delay_prob
    
    # Model 2: Risk
    X_risk = make_feature_vector(features, RISK_FEATURES)
    X_risk_scaled = models['risk_scaler'].transform(X_risk)
    risk_idx = models['risk_model'].predict(X_risk_scaled)[0]
    risk_label = models['risk_le'].inverse_transform([risk_idx])[0]
    risk_probas = models['risk_model'].predict_proba(X_risk_scaled)[0]
    risk_confidence = round(float(max(risk_probas)) * 100, 1)
    
    # Model 3: Fuel
    X_fuel = make_feature_vector(features, FUEL_FEATURES)
    X_fuel_scaled = models['fuel_scaler'].transform(X_fuel)
    optimal_fuel = float(models['fuel_model'].predict(X_fuel_scaled)[0])
    optimal_fuel = max(2.0, optimal_fuel)

    # ── Calculate Route Economics ──
    pickup_distance = get_distance(vehicle_city, origin)
    delivery_distance = get_distance(origin, destination)
    total_distance = pickup_distance + delivery_distance

    fuel_cost_per_km = (optimal_fuel / 100) * 95  # Rs 95/litre
    total_fuel_cost = fuel_cost_per_km * total_distance
    toll_estimate = total_distance * 2.5  # ~Rs 2.5/km average toll
    driver_cost = (total_distance / 300) * 1500  # Rs 1500/day, 300km/day
    total_cost = total_fuel_cost + toll_estimate + driver_cost

    profit = price_inr - total_cost
    profit_margin = (profit / price_inr * 100) if price_inr > 0 else 0
    price_per_km = price_inr / total_distance if total_distance > 0 else 0

    # ── Capacity Check ──
    available_capacity = vehicle_capacity - current_load
    can_carry = available_capacity >= weight_kg

    # ── Decision Logic ──
    reasons_accept = []
    reasons_reject = []
    score = 50  # Start neutral

    # Profitability
    if profit_margin > 25:
        score += 20
        reasons_accept.append(f"High profit margin: {profit_margin:.1f}% (Rs {profit:,.0f} net profit)")
    elif profit_margin > 10:
        score += 10
        reasons_accept.append(f"Decent profit margin: {profit_margin:.1f}% (Rs {profit:,.0f} net profit)")
    elif profit_margin > 0:
        score += 2
        reasons_accept.append(f"Low but positive margin: {profit_margin:.1f}%")
    else:
        score -= 25
        reasons_reject.append(f"LOSS-making route: margin {profit_margin:.1f}% (loss of Rs {abs(profit):,.0f})")

    # Price per km
    if price_per_km > 80:
        score += 10
        reasons_accept.append(f"Excellent rate: Rs {price_per_km:.0f}/km")
    elif price_per_km < 30:
        score -= 10
        reasons_reject.append(f"Below market rate: Rs {price_per_km:.0f}/km (market avg: Rs 50-80/km)")

    # Capacity
    if not can_carry:
        score -= 40
        reasons_reject.append(f"Overweight: load {weight_kg:,} kg exceeds available capacity {available_capacity:,} kg")
    else:
        utilization = (weight_kg / vehicle_capacity) * 100
        if utilization > 70:
            score += 5
            reasons_accept.append(f"Good capacity utilization: {utilization:.0f}%")

    # Delay risk from ML
    if delay_prob > 0.7:
        score -= 15
        reasons_reject.append(f"ML predicts {delay_prob*100:.0f}% delay probability on this route")
    elif delay_prob < 0.3:
        score += 10
        reasons_accept.append(f"Low delay risk: only {delay_prob*100:.0f}% probability")

    # Risk from ML
    if risk_label == 'High Risk':
        score -= 15
        reasons_reject.append(f"ML classifies route as High Risk ({risk_confidence}% confidence)")
    elif risk_label == 'Low Risk':
        score += 10
        reasons_accept.append(f"ML classifies route as Low Risk ({risk_confidence}% confidence)")

    # Fuel level
    if fuel_level < 20:
        score -= 10
        reasons_reject.append(f"Low fuel ({fuel_level}%) - needs refueling before pickup, adds delay")

    # Pickup detour
    if pickup_distance > 300:
        score -= 10
        reasons_reject.append(f"Long pickup detour: {pickup_distance} km from current location ({vehicle_city})")
    elif pickup_distance < 100:
        score += 10
        reasons_accept.append(f"Pickup is nearby: only {pickup_distance} km from {vehicle_city}")

    # Vehicle status
    if vehicle_status == 'maintenance':
        score -= 30
        reasons_reject.append("Vehicle is under maintenance")
    elif vehicle_status == 'delayed':
        score -= 15
        reasons_reject.append("Vehicle already delayed on current assignment")

    # Clamp score
    score = max(0, min(100, score))
    
    # Final decision
    if score >= 55 and can_carry:
        decision = 'ACCEPT'
        summary = f"ACCEPT this load. Profitable route ({origin} to {destination}, {total_distance} km) with Rs {profit:,.0f} estimated profit at {profit_margin:.1f}% margin."
    else:
        decision = 'REJECT'
        top_reason = reasons_reject[0] if reasons_reject else "Overall score too low"
        summary = f"SKIP this load. {top_reason}."

    return jsonify({
        'success': True,
        'decision': decision,
        'confidence_score': score,
        'summary': summary,
        'reasons_accept': reasons_accept,
        'reasons_reject': reasons_reject,
        'economics': {
            'price_inr': price_inr,
            'total_cost_inr': round(total_cost),
            'estimated_profit_inr': round(profit),
            'profit_margin_percent': round(profit_margin, 1),
            'price_per_km': round(price_per_km, 1),
            'pickup_distance_km': pickup_distance,
            'delivery_distance_km': delivery_distance,
            'total_distance_km': total_distance,
            'fuel_cost_inr': round(total_fuel_cost),
            'toll_estimate_inr': round(toll_estimate),
        },
        'ml_analysis': {
            'delay_probability': round(delay_prob, 4),
            'risk_classification': risk_label,
            'risk_confidence': risk_confidence,
            'optimal_fuel_rate': round(optimal_fuel, 2),
        },
        'vehicle': plate,
        'route': f"{origin} -> {destination}",
    })


# ───────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("[*] AI Path Logistics - ML Prediction Server")
    print(f"   Models loaded: {models.get('loaded', False)}")
    print(f"   Starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
