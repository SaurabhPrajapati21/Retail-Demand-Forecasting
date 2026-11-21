# import libraries
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_forecasting_model.pkl"


# load trained model
def load_model():
    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    features = saved_model["features"]

    return model, features

# load sales data
def load_sales_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["store", "item", "date"]).copy()

    return df


# create forecasting features
def create_forecast_features(history, target_date):
    history = history.sort_values("date")

    features = {
        "lag_7": history["sales"].iloc[-7],
        "lag_14": history["sales"].iloc[-14],
        "lag_28": history["sales"].iloc[-28],
        "rolling_mean_7": history["sales"].tail(7).mean(),
        "rolling_mean_28": history["sales"].tail(28).mean(),
        "year": target_date.year,
        "month": target_date.month,
        "day_of_week": target_date.day_of_week,
        "week_of_year": target_date.isocalendar().week,
        "is_weekend": int(target_date.day_of_week >= 5)
    }

    return features


# predict demand for one store and item
def predict_next_day(history, model, features, store, item):
    item_history = history[
        (history["store"] == store) &
        (history["item"] == item)
    ].copy()

    if len(item_history) < 28:
        raise ValueError("At least 28 days of history are required.")

    target_date = item_history["date"].max() + pd.Timedelta(days=1)

    forecast_features = create_forecast_features(
        item_history,
        target_date
    )

    forecast_features["store"] = store
    forecast_features["item"] = item

    input_data = pd.DataFrame(
        [forecast_features]
    )[features]

    prediction = model.predict(input_data)[0]

    return max(0, prediction)



# generates multi-day forecast
def forecast_demand(history, model, features, store, item, days=30):
    item_history = history[
        (history["store"] == store) &
        (history["item"] == item)
    ].copy()

    forecasts = []

    for _ in range(days):
        prediction = predict_next_day(
            item_history,
            model,
            features,
            store,
            item
        )

        next_date = item_history["date"].max() + pd.Timedelta(days=1)

        forecasts.append({
            "date": next_date,
            "store": store,
            "item": item,
            "predicted_sales": prediction
        })

        new_row = pd.DataFrame([{
            "date": next_date,
            "store": store,
            "item": item,
            "sales": prediction
        }])

        item_history = pd.concat(
            [item_history, new_row],
            ignore_index=True
        )

    return pd.DataFrame(forecasts)    


# forecast demand for all store-item combinations
def forecast_all_items(history, model, features, days=30):
    all_forecasts = []

    combinations = history[["store", "item"]].drop_duplicates()

    for _, row in combinations.iterrows():
        forecast = forecast_demand(
            history=history,
            model=model,
            features=features,
            store=row["store"],
            item=row["item"],
            days=days
        )

        all_forecasts.append(forecast)

    return pd.concat(all_forecasts, ignore_index=True)


# test forecasting pipeline
if __name__ == "__main__":
    model, features = load_model()
    history = load_sales_data()

    forecasts = forecast_all_items(
        history=history,
        model=model,
        features=features,
        days=7
    )

    print("\n7-day demand forecast for all items:")
    print("Forecast shape:", forecasts.shape)

    print("\nSample forecasts:")
    print(forecasts.head(10))

    # save forecast results
    forecasts.to_csv(
        BASE_DIR / "data" / "forecast_results.csv",
        index=False
    )

    print("\nForecast results saved successfully.")

    print("\nTotal predicted demand:")
    print(forecasts["predicted_sales"].sum())
     