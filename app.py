"""
VPP Orchestration Engine — Flask Backend (with ML)
====================================================
Run: python app.py
Open: http://localhost:5001
"""

import threading
import time
import json
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from algorithm import VPPOrchestrator, GridState
from simulator import GridSimulator, BatteryFleet
from ml_predictor import PricePredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vpp-secret-2024'
socketio = SocketIO(app, cors_allowed_origins='*')

# ── Global state ───────────────────────────────────────────────────────────────
sim       = GridSimulator()
fleet     = BatteryFleet(n=12)
engine    = VPPOrchestrator(fleet.batteries)
predictor = None   # loaded after first request (takes ~3s to train)
running   = False
sim_thread = None

price_history   = []
revenue_history = []
action_log      = []

def load_predictor():
    global predictor
    print("[ML] Loading price predictor...")
    predictor = PricePredictor()
    print("[ML] Price predictor ready!")

# Train ML model in background on startup
threading.Thread(target=load_predictor, daemon=True).start()


def simulation_loop():
    global running
    while running:
        sim.step(dt_minutes=5)
        state = sim.get_state()

        decisions = engine.decide(state)
        fleet.apply_decisions(decisions, dt_minutes=5)

        fleet_stats    = fleet.get_stats()
        engine_summary = engine.get_fleet_summary()
        weather_info   = sim.get_weather_info()

        price_history.append(round(state.price, 2))
        if len(price_history) > 48: price_history.pop(0)

        revenue_history.append(round(engine.total_revenue, 2))
        if len(revenue_history) > 48: revenue_history.pop(0)

        for dec in decisions:
            if dec.action != 'idle':
                action_log.insert(0, {
                    'time':    sim.get_time_str(),
                    'battery': dec.battery_id,
                    'action':  dec.action,
                    'power':   round(abs(dec.power_kw), 2),
                    'reason':  dec.reason,
                })
        while len(action_log) > 20:
            action_log.pop()

        charges    = sum(1 for d in decisions if d.action == 'charge')
        discharges = sum(1 for d in decisions if d.action == 'discharge')
        idles      = sum(1 for d in decisions if d.action == 'idle')

        # ML predictions
        ml_data = {'predictions': [], 'recommendation': 'HOLD', 'reason': 'Model loading...'}
        if predictor:
            try:
                preds = predictor.predict_next_hours(
                    current_hour=sim.hour,
                    solar=state.solar_forecast,
                    wind=state.wind_forecast,
                    load=state.local_load,
                    n_hours=6
                )
                action, reason = predictor.get_best_action(preds)
                ml_data = {
                    'predictions':    preds,
                    'recommendation': action,
                    'reason':         reason,
                    'mae':            round(predictor.mae, 3),
                }
            except Exception as e:
                print(f"[ML] Prediction error: {e}")

        socketio.emit('update', {
            'time':    sim.get_time_str(),
            'grid': {
                'price':     state.price,
                'solar':     round(state.solar_forecast * 100),
                'wind':      round(state.wind_forecast * 100),
                'load':      state.local_load,
                'frequency': state.frequency,
                'voltage':   state.voltage,
            },
            'weather': {
                'desc':   weather_info.get('desc', 'N/A'),
                'temp':   weather_info.get('temp', 'N/A'),
                'city':   weather_info.get('city', 'N/A'),
                'source': weather_info.get('source', 'sim'),
            },
            'fleet':   fleet_stats,
            'summary': engine_summary,
            'actions': {'charging': charges, 'discharging': discharges, 'idle': idles},
            'price_history':   price_history[-24:],
            'revenue_history': revenue_history[-24:],
            'action_log':      action_log[:10],
            'ml':              ml_data,
        })

        time.sleep(1.5)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify({'running': running, 'time': sim.get_time_str()})


@socketio.on('connect')
def on_connect():
    print('[WS] Client connected')
    emit('status', {'running': running})


@socketio.on('start')
def on_start():
    global running, sim_thread
    if not running:
        running    = True
        sim_thread = threading.Thread(target=simulation_loop, daemon=True)
        sim_thread.start()
        emit('status', {'running': True})


@socketio.on('stop')
def on_stop():
    global running
    running = False
    emit('status', {'running': False})


@socketio.on('reset')
def on_reset():
    global sim, fleet, engine, running, price_history, revenue_history, action_log
    running = False
    time.sleep(0.2)
    sim             = GridSimulator()
    fleet           = BatteryFleet(n=12)
    engine          = VPPOrchestrator(fleet.batteries)
    price_history   = []
    revenue_history = []
    action_log      = []
    emit('status', {'running': False})


if __name__ == '__main__':
    print('=' * 55)
    print('  VPP Orchestration Engine (with ML)')
    print('  Open: http://localhost:5001')
    print('=' * 55)
    import os
port = int(os.environ.get('PORT', 5001))
socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
