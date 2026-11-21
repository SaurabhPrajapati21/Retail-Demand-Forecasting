# Retail Demand Forecasting & Inventory Intelligence

[![Live Demo](https://img.shields.io/badge/Live-Dashboard-brightgreen?style=for-the-badge)](https://retail-demand-ml-dashboard.streamlit.app/)

**Live Application:** [Open Dashboard](https://retail-demand-ml-dashboard.streamlit.app/)

An end-to-end machine learning project that forecasts retail product demand and generates inventory replenishment recommendations.


## Project Overview

This project uses historical retail sales data to predict future demand across multiple stores and products. It combines time-series feature engineering, machine learning, and inventory planning in an interactive Streamlit dashboard.

## Key Features

- Historical sales analysis and visualization
- Demand forecasting using machine learning
- Lag and rolling-window feature engineering
- Chronological model validation
- Inventory reorder point and safety stock calculations
- Replenishment quantity and priority recommendations
- Interactive Streamlit dashboard
- CSV exports for forecasts and inventory recommendations

## Dataset

Store Item Demand Forecasting dataset:

- 913,000 historical daily sales records
- 10 stores
- 50 items
- 5 years of historical data (2013–2017)

## Machine Learning

**Model:** HistGradientBoostingRegressor

**Validation results (2017):**

| Metric | Result |
|---|---:|
| MAE | 6.17 |
| RMSE | 8.02 |

The model was evaluated using a chronological split and rolling one-step-ahead validation.

## Inventory Intelligence

The inventory module calculates:

- Safety stock
- Reorder point
- Target stock
- Suggested replenishment quantity
- Days of inventory cover
- Replenishment priority

Current stock is simulated because actual inventory records are not included in the dataset. Lead time and service-level assumptions are configurable.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Joblib

## Run Locally

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Retail-Demand-Forecasting

python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Project Structure

```text
Retail-Demand-Forecasting/
├── data/
├── notebooks/
├── src/
│   ├── train_model.py
│   ├── forecast.py
│   └── inventory.py
├── models/
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations

- Inventory stock levels are simulated.
- Supplier lead times are assumed.
- Multi-day forecasts are recursive, so prediction errors can accumulate.
- Real operational inventory data would be needed for production deployment.

## Author

Saurabh Prajapati