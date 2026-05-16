"""
VPP Simulator — with Real OpenWeather API
==========================================
Uses real weather data for solar/wind forecast.
Falls back to simulation if API fails.
"""

import random
import math
import time
import requests
from algorithm import BatteryUnit, GridState

# ── Config ────────────────────────────────────────────────────────────────────
NUM_BATTERIES    = 12
OPENWEATHER_KEY  = "c58eb065eb6b0a668bcb38715734c3f6"
CITY             = "Mumbai"   # Change to your city!
WEATHER_INTERVAL = 300        # Fetch real weather every 5 minutes


class WeatherFetcher:
    """Fetches real weather data from OpenWeatherMap API."""

    def __init__(self, api_key, city):
        self.api_key    = api_key
        self.city       = city
        self.last_data  = None
        self.last_fetch = 0

    def get(self):
        now = time.time()
        if now - self.last_fetch < WEATHER_INTERVAL and self.last_data:
            return self.last_data
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}&units=metric"
            r   = requests.get(url, timeout=5)
            d   = r.json()
            clouds     = d.get('clouds', {}).get('all', 50)
            solar      = round(max(0, (100 - clouds) / 100), 3)
            wind_speed = d.get('wind', {}).get('speed', 5)
            wind       = round(min(1.0, wind_speed / 20), 3)
            desc       = d.get('weather', [{}])[0].get('description', 'unknown')
            temp       = d.get('main', {}).get('temp', 25)
            city_name  = d.get('name', self.city)
            self.last_data = {
                'solar': solar, 'wind': wind, 'desc': desc,
                'temp': temp, 'city': city_name, 'clouds': clouds, 'source': 'live',
            }
            self.last_fetch = now
            print(f"[Weather] {city_name}: {desc}, {temp}C, clouds:{clouds}%, solar:{solar}, wind:{wind}")
            return self.last_data
        except Exception as e:
            print(f"[Weather] API failed: {e} — using simulation")
            return self._simulated()

    def _simulated(self):
        return {'solar': random.uniform(0.3, 0.9), 'wind': random.uniform(0.2, 0.7),
                'desc': 'simulated', 'temp': 30, 'city': self.city, 'clouds': 30, 'source': 'sim'}


class GridSimulator:
    def __init__(self):
        self.t       = 0
        self.hour    = 8.0
        self.day     = 0
        self.weather = WeatherFetcher(OPENWEATHER_KEY, CITY)

    def step(self, dt_minutes=5):
        self.t   += 1
        self.hour = (self.hour + dt_minutes / 60) % 24
        if self.hour < dt_minutes / 60:
            self.day += 1

    def get_state(self) -> GridState:
        h  = self.hour
        wx = self.weather.get()
        price_base   = 5.0
        morning_peak = 4.0 * math.exp(-0.5 * ((h - 8.5) / 1.5) ** 2)
        evening_peak = 5.0 * math.exp(-0.5 * ((h - 20)  / 2.0) ** 2)
        night_cheap  = -2.5 if (h < 5 or h > 23) else 0
        solar_effect = -wx['solar'] * 1.5
        noise        = random.gauss(0, 0.3)
        price        = max(1.0, price_base + morning_peak + evening_peak + night_cheap + solar_effect + noise)
        temp_effect  = max(0, (wx['temp'] - 25) * 0.2)
        load_base    = 3.0
        morning_load = 3.0 * math.exp(-0.5 * ((h - 7.5) / 1.5) ** 2)
        evening_load = 5.0 * math.exp(-0.5 * ((h - 19.5) / 2.0) ** 2)
        load         = max(0.5, load_base + morning_load + evening_load + temp_effect + random.gauss(0, 0.4))
        freq_noise   = random.gauss(0, 0.08)
        if load > 9.0:        freq_noise -= 0.3
        if wx['solar'] > 0.9: freq_noise += 0.15
        frequency    = 50.0 + freq_noise
        return GridState(
            price=round(price, 2), solar_forecast=wx['solar'], wind_forecast=wx['wind'],
            local_load=round(load, 2), frequency=round(frequency, 3),
            voltage=round(230.0 + random.gauss(0, 3.0), 1),
        )

    def get_weather_info(self) -> dict:
        return self.weather.last_data or {}

    def get_time_str(self) -> str:
        h = int(self.hour)
        m = int((self.hour - h) * 60)
        return f"Day {self.day + 1}  {h:02d}:{m:02d}"


class BatteryFleet:
    def __init__(self, n=NUM_BATTERIES):
        self.batteries = []
        for i in range(n):
            self.batteries.append(BatteryUnit(
                id=f"BAT-{i+1:03d}", capacity_kwh=random.uniform(5, 15),
                soc=random.uniform(0.3, 0.8), max_charge_kw=random.uniform(3, 7),
                max_discharge_kw=random.uniform(3, 7), online=random.random() > 0.05,
            ))

    def apply_decisions(self, decisions, dt_minutes=5):
        dt_hours = dt_minutes / 60
        bat_map  = {b.id: b for b in self.batteries}
        for dec in decisions:
            b = bat_map.get(dec.battery_id)
            if not b or not b.online: continue
            b.soc = max(0.05, min(1.0, b.soc + (dec.power_kw * dt_hours / b.capacity_kwh)))
            b.soc = max(0.05, b.soc - 0.0001 * random.random())

    def get_stats(self) -> dict:
        online = [b for b in self.batteries if b.online]
        return {
            'units': [{'id': b.id, 'soc': round(b.soc * 100, 1), 'capacity': b.capacity_kwh, 'online': b.online} for b in self.batteries],
            'avg_soc': round(sum(b.soc for b in online) / len(online) * 100, 1) if online else 0,
            'online': len(online), 'total': len(self.batteries),
        }
