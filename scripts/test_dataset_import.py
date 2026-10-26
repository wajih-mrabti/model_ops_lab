from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download the UCI electricity dataset and prepare hourly measurements."""
    print("Loading UCI electricity dataset...")
    dataset = fetch_ucirepo(id=235)

    df = dataset.data.features.copy()

    print("\nOriginal shape:", df.shape)
    print("Columns:", df.columns.tolist())

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

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    print("\nMissing measurement values:")
    print(df[measurement_columns].isna().sum())

    # Average power in kW over each hour approximates energy in kWh.
    hourly = df["Global_active_power"].resample("h").agg(["mean", "count"])
    hourly = hourly.rename(
        columns={
            "mean": "hourly_energy_kwh",
            "count": "valid_minutes",
        }
    )

    # Retain hours with at least 45 valid minute-level readings.
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


if __name__ == "__main__":
    main()
