"""
VPP ML Price Predictor
=======================
Pure numpy implementation — no sklearn needed!
Uses Linear Regression + feature engineering.
Works perfectly on Python 3.13.
"""

import numpy as np
import math
import random


class PricePredictor:
    """
    Predicts electricity prices using Linear Regression (pure numpy).
    Trains on simulated Indian grid historical data.
    """

    def __init__(self):
        self.weights = None
        self.bias    = None
        self.mae     = None
        self._train()

    def _features(self, hour, solar, wind, load):
        """Extract features from inputs."""
        return np.array([
            hour / 24,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(math.pi * hour / 12),
            solar,
            wind,
            load / 12,
            solar * math.sin(math.pi * hour / 12),   # solar * time interaction
            load * (1 - solar),                       # load when solar low
            1 if (hour >= 7 and hour <= 10) else 0,  # morning peak
            1 if (hour >= 18 and hour <= 22) else 0, # evening peak
            1 if (hour < 5 or hour > 23) else 0,     # night cheap
        ])

    def _generate_data(self, n=3000):
        X, y = [], []
        for _ in range(n):
            hour  = random.uniform(0, 24)
            solar = max(0, math.sin(math.pi * (hour - 6) / 12) + random.gauss(0, 0.1)) if 6 <= hour <= 18 else 0
            solar = min(1, max(0, solar))
            wind  = min(1, max(0, 0.3 + random.gauss(0, 0.15)))
            load  = max(0.5, 3.0
                        + 3.0 * math.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)
                        + 5.0 * math.exp(-0.5 * ((hour - 19.5) / 2.0) ** 2)
                        + random.gauss(0, 0.4))

            price = max(1.0,
                5.0
                + 4.0 * math.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2)
                + 5.0 * math.exp(-0.5 * ((hour - 20) / 2.0) ** 2)
                + (-2.5 if (hour < 5 or hour > 23) else 0)
                + (-solar * 1.5)
                + ((load - 5) * 0.3)
                + random.gauss(0, 0.4)
            )
            X.append(self._features(hour, solar, wind, load))
            y.append(price)
        return np.array(X), np.array(y)

    def _train(self):
        print("[ML] Generating training data...")
        X, y = self._generate_data(3000)

        # Normalize
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        Xn = (X - self.X_mean) / self.X_std

        # Linear regression via least squares (pure numpy)
        Xb = np.column_stack([Xn, np.ones(len(Xn))])
        result = np.linalg.lstsq(Xb, y, rcond=None)
        params = result[0]
        self.weights = params[:-1]
        self.bias    = params[-1]

        # MAE on training data
        y_pred   = Xb @ params
        self.mae = float(np.mean(np.abs(y - y_pred)))
        print(f"[ML] Training complete! MAE: Rs{self.mae:.3f}/kWh")

    def _predict_one(self, hour, solar, wind, load):
        f  = self._features(hour, solar, wind, load)
        fn = (f - self.X_mean) / self.X_std
        return float(np.dot(fn, self.weights) + self.bias)

    def predict_next_hours(self, current_hour, solar, wind, load, n_hours=6):
        predictions = []
        for i in range(1, n_hours + 1):
            fh = (current_hour + i) % 24
            fs = max(0, math.sin(math.pi * (fh - 6) / 12) * solar) if 6 <= fh <= 18 else 0
            fl = max(0.5, 3.0
                     + 3.0 * math.exp(-0.5 * ((fh - 7.5) / 1.5) ** 2)
                     + 5.0 * math.exp(-0.5 * ((fh - 19.5) / 2.0) ** 2))
            price = round(max(1.0, self._predict_one(fh, fs, wind * 0.9, fl)), 2)
            conf  = max(60, 95 - i * 5)
            h, m  = int(fh), int((fh - int(fh)) * 60)
            predictions.append({
                'hour':   f"{h:02d}:{m:02d}",
                'price':  price,
                'conf':   conf,
                'action': 'SELL' if price >= 8 else ('BUY' if price <= 3 else 'HOLD')
            })
        return predictions

    def get_best_action(self, predictions):
        if not predictions:
            return 'HOLD', 'No prediction available'
        max_price  = max(p['price'] for p in predictions)
        next_price = predictions[0]['price']
        min_price  = min(p['price'] for p in predictions)
        if max_price >= 8 and next_price < 6:
            return 'CHARGE', f'Price spike expected Rs{max_price:.1f}/kWh — charge now!'
        elif next_price >= 8:
            return 'DISCHARGE', f'High price Rs{next_price:.1f}/kWh — sell to grid!'
        elif min_price <= 3:
            return 'CHARGE', f'Cheap price coming Rs{min_price:.1f}/kWh — buy now!'
        else:
            return 'HOLD', f'Price stable ~Rs{next_price:.1f}/kWh'
