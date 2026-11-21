
# import libraries
import pandas as pd
import numpy as np

from pathlib import Path


# set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
FORECAST_PATH = BASE_DIR / "data" / "forecast_results.csv"
OUTPUT_PATH = BASE_DIR / "data" / "inventory_recommendations.csv"


# load sales and forecast data
def load_data():
    history = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    return history, forecasts


# calculate inventory requirements
def calculate_inventory(
    history,
    forecasts,
    lead_time=7,
    coverage_days=14,
    service_level_z=1.65,
    random_state=42
):
    if lead_time <= 0 or coverage_days <= 0:
        raise ValueError("Lead time and coverage days must be positive.")

    if service_level_z <= 0:
        raise ValueError("Service level factor must be positive.")

    if history.empty or forecasts.empty:
        raise ValueError("Historical sales and forecasts cannot be empty.")

    required_history = {"store", "item", "sales"}
    required_forecasts = {"store", "item", "predicted_sales"}

    if not required_history.issubset(history.columns):
        raise ValueError("Historical data is missing required columns.")

    if not required_forecasts.issubset(forecasts.columns):
        raise ValueError("Forecast data is missing required columns.")

    if forecasts["predicted_sales"].isna().any():
        raise ValueError("Forecast data contains missing predictions.")

    if (forecasts["predicted_sales"] < 0).any():
        raise ValueError("Forecast demand cannot be negative.")

    # calculate average forecast demand
    demand_summary = forecasts.groupby(
        ["store", "item"]
    )["predicted_sales"].mean().reset_index(
        name="avg_daily_demand"
    )

    # calculate historical demand variability
    demand_variability = history.groupby(
        ["store", "item"]
    )["sales"].std().reset_index(
        name="demand_std"
    )

    # combine demand calculations
    inventory = demand_summary.merge(
        demand_variability,
        on=["store", "item"],
        how="left",
        validate="one_to_one"
    )

    inventory["demand_std"] = inventory["demand_std"].fillna(0)

    # simulate current stock because actual stock data is unavailable
    rng = np.random.default_rng(random_state)

    inventory["current_stock"] = rng.integers(
        50,
        1501,
        size=len(inventory)
    )

    # calculate safety stock and inventory thresholds
    inventory["safety_stock"] = (
        service_level_z
        * inventory["demand_std"]
        * np.sqrt(lead_time)
    )

    inventory["reorder_point"] = (
        inventory["avg_daily_demand"] * lead_time
        + inventory["safety_stock"]
    )

    inventory["target_stock"] = (
        inventory["avg_daily_demand"] * (lead_time + coverage_days)
        + inventory["safety_stock"]
    )

    # calculate suggested replenishment quantity
    inventory["reorder_quantity"] = np.where(
    inventory["current_stock"] <= inventory["reorder_point"],
    np.ceil(
        (inventory["target_stock"] - inventory["current_stock"]).clip(lower=0)
    ),
    0
     )

    # classify stock status
    inventory["stock_status"] = np.where(
        inventory["current_stock"] <= inventory["reorder_point"],
        "Reorder Required",
        "Stock Sufficient"
    )

    # calculate days of stock cover
    inventory["days_of_cover"] = np.where(
        inventory["avg_daily_demand"] > 0,
        inventory["current_stock"] / inventory["avg_daily_demand"],
        np.inf
    )

    # assign replenishment priority
    inventory["priority"] = np.select(
        [
            inventory["current_stock"]
            <= inventory["avg_daily_demand"] * lead_time,

            inventory["current_stock"]
            <= inventory["reorder_point"]
        ],
        [
            "Critical",
            "High"
        ],
        default="Normal"
    )

    # sort by replenishment urgency
    inventory = inventory.sort_values(
        ["reorder_quantity", "days_of_cover"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return inventory


# validate inventory calculations
def validate_inventory(inventory):
    checks = {
        "Missing values": inventory.isnull().sum().sum(),
        "Negative reorder quantities": (
            inventory["reorder_quantity"] < 0
        ).sum(),
        "Invalid reorder points": (
            inventory["reorder_point"] < 0
        ).sum(),
        "Incorrect stock classifications": (
            (
                (inventory["current_stock"] <= inventory["reorder_point"])
                & (inventory["stock_status"] != "Reorder Required")
            )
            |
            (
                (inventory["current_stock"] > inventory["reorder_point"])
                & (inventory["stock_status"] != "Stock Sufficient")
            )
        ).sum(),
        "Duplicate store-item combinations": inventory.duplicated(
            ["store", "item"]
        ).sum()
    }

    print("\nInventory validation:")
    for check, result in checks.items():
        print(f"{check}: {result}")

    if any(checks.values()):
        raise ValueError("Inventory validation failed.")

    print("All inventory checks passed.")


# save inventory recommendations
def save_inventory(inventory):
    inventory.to_csv(OUTPUT_PATH, index=False)
    print(f"\nInventory recommendations saved: {OUTPUT_PATH}")


# run inventory pipeline
if __name__ == "__main__":
    history, forecasts = load_data()

    inventory = calculate_inventory(
        history=history,
        forecasts=forecasts,
        lead_time=7,
        coverage_days=14,
        service_level_z=1.65
    )

    validate_inventory(inventory)

    save_inventory(inventory)

    print("\nInventory summary:")
    print("Total combinations:", len(inventory))
    print(
        "Reorder required:",
        (inventory["stock_status"] == "Reorder Required").sum()
    )
    print(
        "Stock sufficient:",
        (inventory["stock_status"] == "Stock Sufficient").sum()
    )
    print(
        "Total suggested reorder quantity:",
        inventory["reorder_quantity"].sum()
    )

    print("\nTop 10 replenishment priorities:")
    print(inventory.head(10))
