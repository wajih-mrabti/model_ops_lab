
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]

print("Loading UCI electricity dataset...")
dataset = fetch_ucirepo(id=235)

# This dataset has no separate target: select one ourselves.
df = dataset.data.features.copy()

print("\nOriginal shape:", df.shape)
print("Columns:", df.columns.tolist())

# Convert measurement columns to numeric.
measurement_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

for column in measurement_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Combine the date and time into one timestamp.
df["timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce",
)

# Remove rows with invalid timestamps.
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

print("\nMissing measurement values:")
print(df[measurement_columns].isna().sum())

# Average power within each hour.
# For a one-hour interval, average kW approximates hourly kWh.
power = df["Global_active_power"]

hourly = power.resample("h").agg(["mean", "count"])
hourly = hourly.rename(
    columns={
        "mean": "hourly_energy_kwh",
        "count": "valid_minutes",
    }
)

# Keep hours with at least 45 valid minute readings.
hourly = hourly[hourly["valid_minutes"] >= 45]
hourly = hourly.dropna(subset=["hourly_energy_kwh"])

output_path = ROOT / "data" / "processed" / "hourly_electricity.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
hourly.to_csv(output_path, index_label="timestamp")

print("\nHourly dataset created!")
print("Output:", output_path)
print("Shape:", hourly.shape)
print("\nFirst rows:")
print(hourly.head())
print("\nLast timestamp:", hourly.index.max())
