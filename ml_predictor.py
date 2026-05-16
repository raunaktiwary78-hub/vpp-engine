"""
VPP ML Price Predictor
=======================
Predicts next-hour electricity price using:
- Time of day (hour)
- Solar forecast
- Wind forecast
- Local load
- Day of week

Uses scikit-learn Random Forest — trains on simulated historical data.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math
import random

MODEL_FILE  = "vpp_model.pkl"
SCALER_FILE = "vpp_scaler.pkl"


def generate_training_data(n_samples=5000):
    """
    Generate realistic historical price data for training.
    Simulates 1 year of Indian grid price patterns.
    """
    X, y = [], []

    for _ in range(n_samples):
        hour       = random.uniform(0, 24)
        day_of_week= random.randint(0, 6)
        solar      = max(0, math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        solar     += random.gauss(0, 0.1)
        solar      = max(0, min(1, solar))
        wind       = max(0, 0.3 + 0.2 * math.sin(2 * math.pi * hour / 24) + random.gauss(0, 0.15))
        wind       = min(1, wind)

        load_base    = 3.0
        morning_load = 3.0 * math.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)
        evening_load = 5.0 * math.exp(-0.5 * ((hour - 19.5) / 2.0) ** 2)
        load         = max(0.5, load_base + morning_load + evening_load + random.gauss(0, 0.4))

        # Weekend effect
        weekend_effect = -0.5 if day_of_week >= 5 else 0

        # Price formula (what model will learn to predict)
        price_base   = 5.0
        morning_peak = 4.0 * math.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2)
        evening_peak = 5.0 * math.exp(-0.5 * ((hour - 20)  / 2.0) ** 2)
        night_cheap  = -2.5 if (hour < 5 or hour > 23) else 0
        solar_effect = -solar * 1.5
        load_effect  = (load - 5) * 0.3
        noise        = random.gauss(0, 0.5)

        price = max(1.0, price_base + morning_peak + evening_peak +
                    night_cheap + solar_effect + load_effect +
                    weekend_effect + noise)

        X.append([hour, math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
                  day_of_week, solar, wind, load])
        y.append(round(price, 2))

    return np.array(X), np.array(y)


def train_model():
    """Train the Random Forest model and save it."""
    print("[ML] Generating training data...")
    X, y = generate_training_data(n_samples=5000)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    print("[ML] Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    print(f"[ML] Training complete! MAE: Rs{mae:.3f}/kWh")

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[ML] Model saved to {MODEL_FILE}")
    return model, scaler, mae


class PricePredictor:
    """
    Predicts electricity prices for the next 6 hours.
    Loads pre-trained model or trains a new one.
    """

    def __init__(self):
        self.model  = None
        self.scaler = None
        self.mae    = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            print("[ML] Loading saved model...")
            with open(MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            with open(SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)
            self.mae = 0.3  # approximate
        else:
            self.model, self.scaler, self.mae = train_model()

    def predict_next_hours(self, current_hour, solar, wind, load, n_hours=6):
        """
        Predict prices for next n_hours.
        Returns list of {hour, predicted_price, confidence}.
        """
        predictions = []

        for i in range(1, n_hours + 1):
            future_hour = (current_hour + i) % 24
            dow         = 0  # assume weekday

            # Future solar estimate
            if 6 <= future_hour <= 18:
                future_solar = max(0, math.sin(math.pi * (future_hour - 6) / 12) * solar)
            else:
                future_solar = 0

            # Future load estimate
            future_load = 3.0 + 3.0 * math.exp(-0.5 * ((future_hour - 7.5)/1.5)**2) + \
                          5.0 * math.exp(-0.5 * ((future_hour - 19.5)/2.0)**2)

            features = np.array([[
                future_hour,
                math.sin(2 * math.pi * future_hour / 24),
                math.cos(2 * math.pi * future_hour / 24),
                dow,
                future_solar,
                wind * 0.9,  # slight wind decay
                future_load
            ]])

            features_scaled = self.scaler.transform(features)
            predicted_price = float(self.model.predict(features_scaled)[0])

            # Confidence decreases for further predictions
            confidence = max(60, 95 - (i * 5))

            h = int(future_hour)
            m = int((future_hour - h) * 60)

            predictions.append({
                'hour':  f"{h:02d}:{m:02d}",
                'price': round(predicted_price, 2),
                'conf':  confidence,
                'action': 'SELL' if predicted_price >= 8 else ('BUY' if predicted_price <= 3 else 'HOLD')
            })

        return predictions

    def get_best_action(self, predictions):
        """Given predictions, what should we do NOW?"""
        if not predictions:
            return 'HOLD', 'No prediction available'

        max_price = max(p['price'] for p in predictions)
        min_price = min(p['price'] for p in predictions)

        # If price will spike soon → charge now to sell later
        next_hour = predictions[0]['price']
        if max_price >= 8 and next_hour < 6:
            return 'CHARGE', f'Price spike expected: Rs{max_price:.1f}/kWh — charge now!'
        elif next_hour >= 8:
            return 'DISCHARGE', f'High price now Rs{next_hour:.1f}/kWh — sell to grid!'
        elif min_price <= 3:
            return 'CHARGE', f'Cheap price coming Rs{min_price:.1f}/kWh — prepare to buy'
        else:
            return 'HOLD', f'Price stable around Rs{next_hour:.1f}/kWh'
