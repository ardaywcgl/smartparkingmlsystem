from flask import Flask, render_template, jsonify, request
import json
import os
import joblib
import pandas as pd

from datetime import datetime, timedelta
import math

app = Flask(__name__)

# Configuration
DATA_FILE = "data.json"
MODEL_PATH = "parking_model.joblib"
META_PATH = "model_metadata.joblib"
CAPACITY = 100

def load_data():
    """Loads system data from data.json. If it doesn't exist, creates with default values."""
    if not os.path.exists(DATA_FILE):
        default_data = {
            "capacity": CAPACITY,
            "current_occupancy": 0,
            "slots": [False] * CAPACITY,
            "events": []
        }
        save_data(default_data)
        return default_data
    
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"Error: Problem reading {DATA_FILE}. Loading default data.")
        return {"capacity": CAPACITY, "current_occupancy": 0, "slots": [False] * CAPACITY, "events": []}


def save_data(data):
    """Saves system data to data.json."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

class PredictionEngine:
    """Hybrid prediction engine: ML, Event Impact, and Real-time Correction."""
    
    def __init__(self):
        self.model = None
        self.metadata = None
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print("Warning: parking_model.joblib not found. Basic predictions will be used.")
            
        if os.path.exists(META_PATH):
            self.metadata = joblib.load(META_PATH)
            print("Model metadata loaded.")
        else:
            print("Warning: model_metadata.joblib not found.")

    def get_base_prediction(self, dt):
        """Predicts base occupancy rate using the machine learning model."""
        if not self.model or not self.metadata:
            return 0.5 * CAPACITY
        
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        is_weekend = 1 if day_of_week >= 5 else 0
        week_of_year = dt.isocalendar()[1]
        
        rolling_3h = self.metadata.get("last_rolling_3h", 0.5)
        rolling_24h = self.metadata.get("last_rolling_24h", 0.5)

        features = pd.DataFrame(
            [[hour, day_of_week, month, is_weekend, week_of_year, rolling_3h, rolling_24h]], 
            columns=self.metadata["features"]
        )
        
        rate = self.model.predict(features)[0]
        return rate * CAPACITY

    def calculate_event_impact(self, target_dt, event_start, event_end, extra_cars):
        """Calculates the impact of events on parking density using Gaussian distribution."""
        start = datetime.strptime(event_start, "%Y-%m-%dT%H:%M")
        end = datetime.strptime(event_end, "%Y-%m-%dT%H:%M")
        
        center = start + (end - start) / 2
        duration_hours = (end - start).total_seconds() / 3600
        diff_hours = (target_dt - center).total_seconds() / 3600
        
        sigma = duration_hours / 4 
        if sigma == 0:
            return 0
        
        impact_factor = math.exp(-(diff_hours**2) / (2 * (sigma**2)))
        
        if abs(diff_hours) > duration_hours:
            return 0
        
        return extra_cars * impact_factor

    def predict(self, target_dt):
        """
        Prediction Engine Logic:
        1. ML Prediction: Analyzes trends in historical data (hour, day, etc.).
        2. Event Impact: Calculates event density using Gaussian (bell curve) distribution.
        3. Exponential Smoothing: Applies the difference between current parking status and prediction with damping.
        """
        data = load_data()
        now = datetime.now()
        
        # 1. Base ML Prediction
        base_pred = self.get_base_prediction(target_dt)
        reasons = []
        
        # 2. Event Impact (Gaussian Density Calculation)
        event_impact = 0
        for event in data["events"]:
            impact = self.calculate_event_impact(target_dt, event["start"], event["end"], event["extra_cars"])
            if impact > 1:
                event_impact += impact
                intensity = int((impact / event['extra_cars']) * 100)
                reasons.append(f"Event Impact (Density: {intensity}%)")

        prediction = base_pred + event_impact

        # 3. Real-time Exponential Smoothing (Exponential Decay)
        hours_diff = abs((target_dt - now).total_seconds()) / 3600
        k = 0.7  # Damping coefficient (rate of effect decay over time)
        decay = math.exp(-k * hours_diff)
        
        current_occ = data["current_occupancy"]
        expected_now = self.get_base_prediction(now)
        delta = current_occ - expected_now
        
        adj_value = delta * decay
        prediction += adj_value

        
        if abs(adj_value) > 2:
            reasons.append(f"Instant Data Correction (Impact: {decay:.2f})")

        # 4. Limitation (0 - Capacity)
        prediction = max(0, min(CAPACITY, prediction))
        occupancy_rate = (prediction / CAPACITY) * 100
        
        level = "Available" if occupancy_rate < 40 else "Moderate" if occupancy_rate < 80 else "Busy"
        
        return {
            "predicted_occupancy": round(prediction, 1),
            "occupancy_percentage": round(occupancy_rate, 1),
            "available_spaces": round(CAPACITY - prediction, 1),
            "level": level,
            "reasons": reasons if reasons else ["Normal traffic flow"],
            "performance_info": f"Model MAE: {round(self.metadata.get('mae', 0)*100, 2)}%" if self.metadata else "Model not trained"
        }

engine = PredictionEngine()

# --- Route Definitions ---

@app.route('/')
def index():
    """User Panel."""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Admin Panel."""
    return render_template('admin.html')

@app.route('/api/status')
def get_status():
    """Returns the current status of the parking lot."""
    return jsonify(load_data())

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Makes a prediction for a specific date and time."""
    dt_str = request.json.get("datetime")
    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
    return jsonify(engine.predict(dt))

@app.route('/api/forecast')
def api_forecast():
    """Returns an hourly prediction list for the next 12 hours."""
    results = []
    now = datetime.now()
    for i in range(12):
        future = now + timedelta(hours=i)
        res = engine.predict(future)
        results.append({
            "time": future.strftime("%H:00"), 
            "occupancy": res["predicted_occupancy"]
        })
    return jsonify(results)

# --- Admin API Operations ---

@app.route('/api/admin/occupancy', methods=['POST'])
def update_occupancy():
    """Manual occupancy increment/decrement."""
    data = load_data()
    action = request.json.get("action")
    if action == "inc":
        data["current_occupancy"] = min(CAPACITY, data["current_occupancy"] + 1)
    else:
        data["current_occupancy"] = max(0, data["current_occupancy"] - 1)
    save_data(data)
    return jsonify(data)

@app.route('/api/admin/slot', methods=['POST'])
def toggle_slot():
    """Reserve/free a specific parking slot."""
    data = load_data()
    idx = request.json.get("index")
    data["slots"][idx] = not data["slots"][idx]
    data["current_occupancy"] = sum(data["slots"])
    save_data(data)
    return jsonify(data)

@app.route('/api/admin/event', methods=['POST'])
def add_event():
    """Adds a new event to the system."""
    data = load_data()
    data["events"].append(request.json)
    save_data(data)
    return jsonify(data)

@app.route('/api/admin/event/<int:index>', methods=['DELETE'])
def delete_event(index):
    """Deletes an event from the system."""
    data = load_data()
    data["events"].pop(index)
    save_data(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

