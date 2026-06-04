"""
AV Accident Prediction — Flask Backend
--------------------------------------
Run:  python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import json
import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Load saved models ──────────────────────────────────────────────
MODEL_DIR = "saved_models"

rf_model      = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
mlp_model     = tf.keras.models.load_model(os.path.join(MODEL_DIR, "mlp_model.keras"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

with open(os.path.join(MODEL_DIR, "ensemble_config.json")) as f:
    config = json.load(f)

THRESHOLD = config["threshold"]
print(f"✅  Models loaded | {len(feature_names)} features | threshold={THRESHOLD}")


# ── Feature engineering (must mirror training notebook) ────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speed_ratio"]      = df["speed_kmh"] / 120
    df["safety_margin"]    = df["following_dist_m"] / (df["speed_kmh"] / 3.6 + 1e-6)
    df["visibility_ratio"] = df["visibility_m"] / 300
    df["composite_risk"]   = df["road_wetness"] * (1 / (df["visibility_m"] + 1)) * df["speed_kmh"]
    df["congestion_index"] = df["traffic_density"] * (df["pedestrian_count"] + 1)
    df["sensor_penalty"]   = 1 - df["sensor_health"]
    df = pd.get_dummies(df, columns=["time_of_day", "road_type", "weather"], drop_first=False)
    return df


def predict(sample: dict) -> dict:
    df_s = pd.DataFrame([sample])
    df_s = engineer_features(df_s.assign(accident=0)).drop("accident", axis=1)
    for col in feature_names:
        if col not in df_s.columns:
            df_s[col] = 0
    df_s = df_s[feature_names]
    X = scaler.transform(df_s)

    p_rf  = float(rf_model.predict_proba(X)[0, 1])
    p_mlp = float(mlp_model.predict(X, verbose=0)[0, 0])
    p_ens = 0.5 * p_rf + 0.5 * p_mlp
    label = "ACCIDENT" if p_ens >= THRESHOLD else "SAFE"

    # Per-feature risk breakdown
    feature_risk = {
        "Visibility":       round(max(0, 1 - sample["visibility_m"] / 250), 3),
        "Speed":            round(max(0, (sample["speed_kmh"] - 50) / 130), 3),
        "Following dist.":  round(max(0, 1 - sample["following_dist_m"] / 40), 3),
        "Road wetness":     round(sample["road_wetness"], 3),
        "Sensor health":    round(1 - sample["sensor_health"], 3),
        "Traffic density":  round(sample["traffic_density"] / 10, 3),
        "Pedestrians":      round(sample["pedestrian_count"] / 20, 3),
        "Lane changes":     round(sample["lane_change_rate"] / 15, 3),
    }

    return {
        "prediction":    label,
        "probability":   round(p_ens, 4),
        "rf_prob":       round(p_rf, 4),
        "mlp_prob":      round(p_mlp, 4),
        "feature_risks": feature_risk,
    }


# ── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    required = [
        "speed_kmh", "following_dist_m", "visibility_m", "road_wetness",
        "pedestrian_count", "lane_change_rate", "sensor_health",
        "traffic_density", "steering_deviation",
        "time_of_day", "road_type", "weather",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = predict(data)
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "threshold": THRESHOLD, "features": len(feature_names)})


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
