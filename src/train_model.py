# import libraries
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# load sales data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


# create forecasting features
def create_features(df):
    df = df.sort_values(["store", "item", "date"]).copy()

    groups = df.groupby(["store", "item"])["sales"]

    df["lag_7"] = groups.shift(7)
    df["lag_14"] = groups.shift(14)
    df["lag_28"] = groups.shift(28)

    df["rolling_mean_7"] = groups.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["rolling_mean_28"] = groups.transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_of_week
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


# prepare modeling dataset
FEATURES = [
    "store", "item",
    "day_of_week", "month", "year",
    "week_of_year", "is_weekend",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28"
]

def prepare_modeling_data(df):
    df = create_features(df)

    df = df.dropna(subset=FEATURES + ["sales"]).copy()

    return df


# train forecasting model
def train_model(df):
    train_data = df[df["date"] < "2017-01-01"]
    valid_data = df[df["date"] >= "2017-01-01"]

    X_train = train_data[FEATURES]
    y_train = train_data["sales"]

    X_valid = valid_data[FEATURES]
    y_valid = valid_data["sales"]

    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
  
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")



    # train final model on all available data
    final_model = HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.1,
        random_state=42
    )

    final_model.fit(df[FEATURES], df["sales"])

    return final_model


def save_model(model):
    model_path = MODEL_DIR / "demand_forecasting_model.pkl"

    joblib.dump(
        {
            "model": model,
            "features": FEATURES
        },
        model_path
    )

    print(f"Model saved: {model_path}")



 # run training pipeline
if __name__ == "__main__":
    sales_df = load_data()

    model_df = prepare_modeling_data(sales_df)

    model = train_model(model_df)

    save_model(model)



       