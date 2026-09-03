import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "simulated_pune_daily_waste_forecasting.csv"
MODELS_DIR = PROJECT_DIR / "forecast_models"

MODEL_PATH = MODELS_DIR / "pune_waste_forecast_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
VALIDATION_RESULTS_PATH = MODELS_DIR / "validation_results.csv"

TRAIN_SPLIT = 0.80
RANDOM_STATE = 42


def add_date_features(dataframe):
    """Create date-based forecasting features."""

    dataframe = dataframe.copy()

    # Your CSV dates use DD-MM-YYYY, for example 13-01-2023.
    dataframe["date"] = pd.to_datetime(
        dataframe["date"].astype(str).str.strip(),
        format="%d-%m-%Y",
        errors="coerce",
    )

    invalid_dates = dataframe["date"].isna()

    if invalid_dates.any():
        invalid_values = dataframe.loc[
            invalid_dates,
            "date",
        ].head(5).tolist()

        raise ValueError(
            "Invalid date values in the CSV: "
            + str(invalid_values)
        )

    dataframe["day_of_week_number"] = dataframe["date"].dt.dayofweek
    dataframe["day_of_year"] = dataframe["date"].dt.dayofyear
    dataframe["week_of_year"] = (
        dataframe["date"].dt.isocalendar().week.astype(int)
    )
    dataframe["year"] = dataframe["date"].dt.year

    return dataframe


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Put the CSV file in the same folder as this script."
        )

    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading Pune waste dataset...")

    # CSV UTF-8 encoding from Excel
    dataframe = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

    required_columns = [
        "date",
        "ward_name",
        "households",
        "segregated_households",
        "baseline_waste_tonnes_per_day",
        "month",
        "is_weekend",
        "is_festival",
        "daily_waste_tonnes",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing these required columns:\n"
            + ", ".join(missing_columns)
        )

    dataframe = add_date_features(dataframe)
    dataframe["date"] = pd.to_datetime(
    dataframe["date"],
    format="%d-%m-%Y",
)

    unique_dates = sorted(dataframe["date"].unique())
    split_index = int(len(unique_dates) * TRAIN_SPLIT)
    split_date = unique_dates[split_index]

    train_data = dataframe[dataframe["date"] < split_date].copy()
    validation_data = dataframe[dataframe["date"] >= split_date].copy()

    print(f"\nTraining records: {len(train_data):,}")
    print(f"Validation records: {len(validation_data):,}")
    print(f"Validation begins: {pd.Timestamp(split_date).date()}")

    feature_columns = [
        "ward_name",
        "households",
        "segregated_households",
        "baseline_waste_tonnes_per_day",
        "day_of_week_number",
        "day_of_year",
        "week_of_year",
        "month",
        "year",
        "is_weekend",
        "is_festival",
    ]

    target_column = "daily_waste_tonnes"

    categorical_features = [
        "ward_name",
    ]

    numerical_features = [
        "households",
        "segregated_households",
        "baseline_waste_tonnes_per_day",
        "day_of_week_number",
        "day_of_year",
        "week_of_year",
        "month",
        "year",
        "is_weekend",
        "is_festival",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numerical_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\nTraining forecast model...")

    pipeline.fit(
        train_data[feature_columns],
        train_data[target_column],
    )

    predictions = pipeline.predict(
        validation_data[feature_columns]
    )

    actual_values = validation_data[target_column]

    mae = mean_absolute_error(actual_values, predictions)
    rmse = mean_squared_error(actual_values, predictions) ** 0.5
    r2 = r2_score(actual_values, predictions)

    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"MAE:  {mae:.2f} tonnes")
    print(f"RMSE: {rmse:.2f} tonnes")
    print(f"R²:   {r2:.4f}")
    print("=" * 55)

    joblib.dump(pipeline, MODEL_PATH)

    validation_output = validation_data[
        [
            "date",
            "ward_name",
            "baseline_waste_tonnes_per_day",
            "is_festival",
            "daily_waste_tonnes",
        ]
    ].copy()

    validation_output.rename(
        columns={
            "daily_waste_tonnes": "actual_waste_tonnes",
        },
        inplace=True,
    )

    validation_output["predicted_waste_tonnes"] = predictions
    validation_output["absolute_error_tonnes"] = (
        validation_output["actual_waste_tonnes"]
        - validation_output["predicted_waste_tonnes"]
    ).abs()

    validation_output.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
    )

    metadata = {
        "dataset": DATASET_PATH.name,
        "target": target_column,
        "model": "RandomForestRegressor",
        "training_records": int(len(train_data)),
        "validation_records": int(len(validation_data)),
        "validation_start_date": str(pd.Timestamp(split_date).date()),
        "mae_tonnes": round(float(mae), 4),
        "rmse_tonnes": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "note": (
            "Prototype trained on simulated daily waste data derived "
            "from Pune ward-level averages."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining completed successfully.")
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Validation results: {VALIDATION_RESULTS_PATH}")
    print(f"Model details: {METADATA_PATH}")


if __name__ == "__main__":
    main()