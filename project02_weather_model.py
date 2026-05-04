
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


LAT = 41.6611
LON = -91.5302
TIMEZONE = "America/Chicago"
OUTPUT_CSV = Path("data/iowa_city_weather_dataset.csv")


def fetch_daily_weather(start_date: str = "2015-01-01", end_date: str | None = None) -> pd.DataFrame:
    """download daily weather data from Open-Meteo archive API"""
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "windspeed_10m_max",
            "windgusts_10m_max",
            "weather_code",
        ],
        "timezone": TIMEZONE,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    raw = response.json()["daily"]
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date").sort_index()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-series features and targets for regression + classification."""
    data = df.copy()

    # Lag features
    for col in ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max"]:
        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag3"] = data[col].shift(3)

    # Rolling features
    data["temp_max_roll7"] = data["temperature_2m_max"].rolling(7).mean()
    data["precip_roll7"] = data["precipitation_sum"].rolling(7).sum()

    # Calendar features
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear

    # Targets: next-day max temp and next-day storm day
    data["target_temp_max_next_day"] = data["temperature_2m_max"].shift(-1)
    data["is_storm_day"] = data["weather_code"].isin([95, 96, 99]).astype(int)
    data["target_storm_next_day"] = data["is_storm_day"].shift(-1)

    # Drop rows with NaN from rolling/lags/shift
    data = data.dropna().copy()

    return data


def train_regression(data: pd.DataFrame) -> dict[str, float]:
    """Train baseline regressor for next-day max temperature."""
    features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "temperature_2m_max_lag1",
        "temperature_2m_max_lag3",
        "temp_max_roll7",
        "month",
        "day_of_year",
    ]

    X = data[features]
    y = data["target_temp_max_next_day"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_classification(data: pd.DataFrame) -> dict[str, float]:
    """train baseline classifier for next-day storm event"""
    features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "precipitation_sum_lag1",
        "precipitation_sum_lag3",
        "precip_roll7",
        "month",
    ]

    X = data[features]
    y = data["target_storm_next_day"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    return {"accuracy": acc, "f1": f1}


def main() -> None:
    print("fetching Iowa City weather data")
    raw = fetch_daily_weather(start_date="2015-01-01")
    data = engineer_features(raw)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_CSV)

    print(f"Saved dataset to: {OUTPUT_CSV.resolve()}")
    print(f"Dataset rows: {len(data):,}")

    reg_metrics = train_regression(data)
    clf_metrics = train_classification(data)

    print("\nRegression (next-day max temp)")
    print(f"RMSE: {reg_metrics['rmse']:.3f}")
    print(f"MAE:  {reg_metrics['mae']:.3f}")
    print(f"R2:   {reg_metrics['r2']:.3f}")

    print("\nClassification (next-day storm)")
    print(f"Accuracy: {clf_metrics['accuracy']:.3f}")
    print(f"F1:       {clf_metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
