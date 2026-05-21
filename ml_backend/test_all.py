import urllib.request
import json

# Test 1: Health Check
print("=" * 50)
print("TEST 1: Health Check")
req = urllib.request.Request("http://localhost:5000/api/health")
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
print(f"  Status: {data['status']}")
print(f"  Models loaded: {data['models_loaded']}")
print(f"  PASS" if data['models_loaded'] else "  FAIL")

# Test 2: Decision Endpoint
print("\n" + "=" * 50)
print("TEST 2: Generate AI Insights (Decision)")
body = json.dumps({"vehicles": [
    {"id": "v-001", "plate_number": "MH-02-CK-4521", "model": "Tata Prima", "status": "moving", "fuel_level": 72, "load_percentage": 80, "city": "Mumbai"},
    {"id": "v-004", "plate_number": "TN-09-AB-1199", "model": "Eicher Pro", "status": "idle", "fuel_level": 91, "load_percentage": 0, "city": "Chennai"},
]}).encode()
req = urllib.request.Request("http://localhost:5000/api/predict/decision", data=body, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
print(f"  Success: {data['success']}")
print(f"  Decisions count: {len(data['decisions'])}")
for d in data['decisions']:
    print(f"    - {d['decision_type']}: {d['reasoning'][:80]}...")
print(f"  PASS" if data['success'] and len(data['decisions']) > 0 else "  FAIL")

# Test 3: Load Decision (Accept/Reject)
print("\n" + "=" * 50)
print("TEST 3: Load Accept/Reject Evaluation")
body = json.dumps({
    "vehicle": {"id": "v-001", "plate_number": "MH-02-CK-4521", "model": "Tata Prima", "status": "idle", "fuel_level": 72, "capacity_kg": 28000, "current_load_kg": 0, "city": "Mumbai"},
    "load": {"origin": "Mumbai", "destination": "Delhi", "weight_kg": 22000, "price_inr": 185000}
}).encode()
req = urllib.request.Request("http://localhost:5000/api/predict/load-decision", data=body, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
print(f"  Decision: {data['decision']}")
print(f"  Score: {data['confidence_score']}/100")
print(f"  Summary: {data['summary']}")
print(f"  Profit: Rs {data['economics']['estimated_profit_inr']:,}")
print(f"  Distance: {data['economics']['total_distance_km']} km")
print(f"  Delay Risk: {data['ml_analysis']['delay_probability']*100:.0f}%")
print(f"  Risk Level: {data['ml_analysis']['risk_classification']}")
print(f"  PASS" if data['success'] else "  FAIL")

# Test 4: Load Decision (Should Reject - overweight)
print("\n" + "=" * 50)
print("TEST 4: Load Reject - Overweight")
body = json.dumps({
    "vehicle": {"id": "v-007", "plate_number": "UP-32-KL-2288", "model": "Ashok Leyland", "status": "delayed", "fuel_level": 18, "capacity_kg": 15000, "current_load_kg": 14200, "city": "Lucknow"},
    "load": {"origin": "Jaipur", "destination": "Chennai", "weight_kg": 30000, "price_inr": 50000}
}).encode()
req = urllib.request.Request("http://localhost:5000/api/predict/load-decision", data=body, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
print(f"  Decision: {data['decision']}")
print(f"  Score: {data['confidence_score']}/100")
print(f"  Summary: {data['summary'][:100]}")
print(f"  PASS" if data['decision'] == 'REJECT' else "  FAIL (expected REJECT)")

print("\n" + "=" * 50)
print("ALL API TESTS COMPLETE")
