
# import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# configure dashboard
st.set_page_config(
    page_title="Retail Demand Intelligence",
    page_icon="📊",
    layout="wide"
)


# set project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SALES_PATH = DATA_DIR / "train.csv"
FORECAST_PATH = DATA_DIR / "forecast_results.csv"
INVENTORY_PATH = DATA_DIR / "inventory_recommendations.csv"


# load project data
@st.cache_data
def load_data():
    sales = pd.read_csv(SALES_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    inventory = pd.read_csv(INVENTORY_PATH)

    return sales, forecasts, inventory


# check required files
missing_files = [
    path.name
    for path in [SALES_PATH, FORECAST_PATH, INVENTORY_PATH]
    if not path.exists()
]

if missing_files:
    st.error(
        "Required data files are missing: "
        + ", ".join(missing_files)
    )
    st.info("Run the forecasting and inventory scripts before starting the dashboard.")
    st.stop()


sales, forecasts, inventory = load_data()


# dashboard title
st.title("Retail Demand Intelligence Dashboard")
st.caption(
    "Demand forecasting, sales analytics and inventory replenishment intelligence"
)


# sidebar filters
st.sidebar.header("Dashboard Filters")

stores = sorted(sales["store"].unique().tolist())
items = sorted(sales["item"].unique().tolist())

selected_store = st.sidebar.selectbox(
    "Select Store",
    ["All"] + stores
)

selected_item = st.sidebar.selectbox(
    "Select Item",
    ["All"] + items
)

min_date = sales["date"].min().date()
max_date = sales["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Historical Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# filter historical sales
filtered_sales = sales.copy()
filtered_forecasts = forecasts.copy()
filtered_inventory = inventory.copy()

if selected_store != "All":
    filtered_sales = filtered_sales[
        filtered_sales["store"] == selected_store
    ]

    filtered_forecasts = filtered_forecasts[
        filtered_forecasts["store"] == selected_store
    ]

    filtered_inventory = filtered_inventory[
        filtered_inventory["store"] == selected_store
    ]

if selected_item != "All":
    filtered_sales = filtered_sales[
        filtered_sales["item"] == selected_item
    ]

    filtered_forecasts = filtered_forecasts[
        filtered_forecasts["item"] == selected_item
    ]

    filtered_inventory = filtered_inventory[
        filtered_inventory["item"] == selected_item
    ]

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_sales = filtered_sales[
        filtered_sales["date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )
    ]


# overview metrics
st.header("Business Overview")

total_sales = filtered_sales["sales"].sum()

average_daily_sales = (
    filtered_sales.groupby("date")["sales"].sum().mean()
    if not filtered_sales.empty else 0
)

total_forecast = filtered_forecasts["predicted_sales"].sum()

reorder_count = (
    filtered_inventory["stock_status"] == "Reorder Required"
).sum()

total_reorder_quantity = filtered_inventory["reorder_quantity"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Historical Sales",
    f"{total_sales:,.0f} units"
)

col2.metric(
    "Average Daily Sales",
    f"{average_daily_sales:,.1f} units"
)

col3.metric(
    "7-Day Forecast",
    f"{total_forecast:,.0f} units"
)

col4.metric(
    "Items Requiring Reorder",
    f"{reorder_count:,}"
)

st.divider()


# model performance
st.header("Forecast Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Model", "HistGradientBoosting")
col2.metric("Validation MAE", "6.17")
col3.metric("Validation RMSE", "8.02")

st.caption(
    "Validation results are based on the 2017 chronological holdout "
    "using rolling one-day-ahead evaluation."
)


# sales analysis
st.header("Historical Sales Analysis")

if filtered_sales.empty:
    st.warning("No historical sales data available for these filters.")
else:
    daily_sales = filtered_sales.groupby(
        "date", as_index=False
    )["sales"].sum()

    monthly_sales = filtered_sales.copy()
    monthly_sales["month"] = (
        monthly_sales["date"].dt.to_period("M").astype(str)
    )

    monthly_sales = monthly_sales.groupby(
        "month", as_index=False
    )["sales"].sum()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            daily_sales,
            x="date",
            y="sales",
            title="Daily Sales Trend",
            labels={
                "date": "Date",
                "sales": "Units Sold"
            }
        )

        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            monthly_sales,
            x="month",
            y="sales",
            title="Monthly Sales Performance",
            labels={
                "month": "Month",
                "sales": "Units Sold"
            }
        )

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        store_performance = filtered_sales.groupby(
            "store", as_index=False
        )["sales"].sum().sort_values(
            "sales", ascending=False
        )

        fig = px.bar(
            store_performance,
            x="store",
            y="sales",
            title="Sales by Store",
            labels={
                "store": "Store",
                "sales": "Total Sales"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        item_performance = filtered_sales.groupby(
            "item", as_index=False
        )["sales"].sum().sort_values(
            "sales", ascending=False
        ).head(10)

        fig = px.bar(
            item_performance,
            x="item",
            y="sales",
            title="Top 10 Items by Sales",
            labels={
                "item": "Item",
                "sales": "Total Sales"
            }
        )

        st.plotly_chart(fig, use_container_width=True)


st.divider()


# demand forecasting
st.header("Demand Forecasting")

if filtered_forecasts.empty:
    st.warning("No forecast data available for these filters.")
else:
    forecast_summary = filtered_forecasts.groupby(
        "date", as_index=False
    )["predicted_sales"].sum()

    fig = px.line(
        forecast_summary,
        x="date",
        y="predicted_sales",
        markers=True,
        title="7-Day Future Demand Forecast",
        labels={
            "date": "Forecast Date",
            "predicted_sales": "Predicted Units"
        }
    )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Details")

    st.dataframe(
    filtered_forecasts.sort_values(
        ["date", "store", "item"]
    ),
    use_container_width=True,
    hide_index=True
)

    st.download_button(
        "Download Forecast CSV",
        data=filtered_forecasts.to_csv(index=False).encode("utf-8"),
        file_name="forecast_results.csv",
        mime="text/csv"
    )


st.divider()


# inventory intelligence
st.header("Inventory Intelligence")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Reorder Required",
    f"{reorder_count:,}"
)

col2.metric(
    "Stock Sufficient",
    f"{len(filtered_inventory) - reorder_count:,}"
)

col3.metric(
    "Suggested Reorder Quantity",
    f"{total_reorder_quantity:,.0f} units"
)

if not filtered_inventory.empty:
    col1, col2 = st.columns(2)

    with col1:
        stock_distribution = filtered_inventory[
            "stock_status"
        ].value_counts().reset_index()

        stock_distribution.columns = [
            "stock_status",
            "count"
        ]

        fig = px.pie(
            stock_distribution,
            names="stock_status",
            values="count",
            title="Inventory Status Distribution",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        priority_distribution = filtered_inventory[
            "priority"
        ].value_counts().reset_index()

        priority_distribution.columns = [
            "priority",
            "count"
        ]

        fig = px.bar(
            priority_distribution,
            x="priority",
            y="count",
            title="Replenishment Priority Distribution",
            labels={
                "priority": "Priority",
                "count": "Store-Item Combinations"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Replenishment Priorities")

    priority_order = {
        "Critical": 0,
        "High": 1,
        "Normal": 2
    }

    priority_table = filtered_inventory.copy()

    priority_table["priority_order"] = (
        priority_table["priority"].map(priority_order)
    )

    priority_table = priority_table.sort_values(
        ["priority_order", "reorder_quantity"],
        ascending=[True, False]
    )

    st.dataframe(
        priority_table.drop(columns=["priority_order"]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Inventory Recommendations",
        data=filtered_inventory.to_csv(index=False).encode("utf-8"),
        file_name="inventory_recommendations.csv",
        mime="text/csv"
    )
else:
    st.warning("No inventory records available for these filters.")


# project assumptions
st.divider()

with st.expander("Project Methodology and Assumptions"):
    st.markdown("""
    **Forecasting**
    - Model: HistGradientBoostingRegressor
    - Training data: 2013–2017 historical sales
    - Validation period: 2017
    - Evaluation: Rolling one-day-ahead predictions
    - Dashboard forecast: Recursive 7-day predictions

    **Inventory planning**
    - Lead time: 7 days
    - Coverage period: 14 days
    - Service-level factor: 1.65
    - Current inventory: Simulated values
    - Supplier lead time: Assumed

    **Limitations**

    Actual inventory levels, supplier information and purchase orders
    are not available in the dataset. Inventory recommendations are
    scenario-based estimates and should not be treated as actual
    purchasing instructions.
    """)

st.caption(
    "Retail Demand Intelligence | Demand Forecasting & Inventory Planning"
)
