import argparse
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "simulated_pune_daily_waste_forecasting.csv"

MODEL_PATH = (
    PROJECT_DIR
    / "forecast_models"
    / "pune_waste_forecast_model.joblib"
)


def create_features(ward_data, prediction_date, is_festival):
    """Create the same features used during model training."""

    prediction_date = pd.Timestamp(prediction_date)

    return pd.DataFrame(
        [
            {
                "ward_name": ward_data["ward_name"],
                "households": ward_data["households"],
                "segregated_households": ward_data[
                    "segregated_households"
                ],
                "baseline_waste_tonnes_per_day": ward_data[
                    "baseline_waste_tonnes_per_day"
                ],
                "day_of_week_number": prediction_date.dayofweek,
                "day_of_year": prediction_date.dayofyear,
                "week_of_year": int(prediction_date.isocalendar().week),
                "month": prediction_date.month,
                "year": prediction_date.year,
                "is_weekend": int(prediction_date.dayofweek >= 5),
                "is_festival": int(is_festival),
            }
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict future waste quantity for a Pune ward."
    )

    parser.add_argument(
        "--ward",
        required=True,
        help='Ward name, for example: "Viman Nagar"',
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of future days to predict. Default: 7",
    )

    parser.add_argument(
        "--festival",
        action="store_true",
        help="Apply a festival scenario to every forecast day.",
    )

    arguments = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Forecast model not found. Run train_forecast_model.py first."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Put the CSV file in this folder."
        )

    if not 1 <= arguments.days <= 30:
        raise ValueError("Choose a value from 1 to 30 for --days.")

    print("\nLoading forecast model...")
    model = joblib.load(MODEL_PATH)

    dataset = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

    ward_matches = dataset[
        dataset["ward_name"].astype(str).str.casefold()
        == arguments.ward.casefold()
    ]

    if ward_matches.empty:
        print("\nWard not found. Available wards:\n")

        for ward in sorted(dataset["ward_name"].unique()):
            print(f"- {ward}")

        return

    ward_data = ward_matches.iloc[0]

    latest_date = pd.to_datetime(
    dataset["date"],
    format="%d-%m-%Y",
).max()
    start_date = latest_date + timedelta(days=1)

    forecast_rows = []

    for day_number in range(arguments.days):
        prediction_date = start_date + timedelta(days=day_number)

        input_data = create_features(
            ward_data=ward_data,
            prediction_date=prediction_date,
            is_festival=arguments.festival,
        )

        predicted_tonnes = float(model.predict(input_data)[0])

        forecast_rows.append(
            {
                "date": prediction_date.strftime("%Y-%m-%d"),
                "predicted_waste_tonnes": round(predicted_tonnes, 2),
            }
        )

    forecast_dataframe = pd.DataFrame(forecast_rows)

    print("\n" + "=" * 60)
    print("PUNE WASTE FORECAST")
    print("=" * 60)

    print(f"\nWard: {ward_data['ward_name']}")
    print(f"Forecast days: {arguments.days}")
    print(
        f"Festival scenario: "
        f"{'Enabled' if arguments.festival else 'Disabled'}"
    )

    print("\nDaily Predictions:")

    for _, row in forecast_dataframe.iterrows():
        print(
            f"{row['date']}  |  "
            f"{row['predicted_waste_tonnes']:.2f} tonnes"
        )

    print("\n" + "-" * 60)
    print(
        f"Total expected waste: "
        f"{forecast_dataframe['predicted_waste_tonnes'].sum():.2f} tonnes"
    )
    print(
        f"Average daily waste:  "
        f"{forecast_dataframe['predicted_waste_tonnes'].mean():.2f} tonnes"
    )
    print("-" * 60)

   


if __name__ == "__main__":
    main()