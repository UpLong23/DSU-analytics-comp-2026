# DSU Analytics Competition 2026

## Overview

This repository contains exploratory analysis, clustering work, and forecasting experiments built around an emergency department encounter dataset. The project appears to focus on two main tasks:

1. Grouping and clustering visit reasons to understand demand patterns.
2. Forecasting encounter volume at different time horizons, including hourly, daily, weekly, and monthly views.

The repository is organized around notebooks for analysis and model development, a shared `utils.py` module, and a `data/` directory that stores both the original dataset and processed outputs.

## Repository structure

```text
.
├── clustering/
│   └── clustering.ipynb
├── data/
│   ├── DSU-Dataset.csv
│   ├── forecast.csv
│   ├── forecast_sep_oct.csv
│   ├── grouped_by_hour_blocks.csv
│   ├── grouped_by_site_hour_block.csv
│   ├── grouped_by_year_month.csv
│   ├── grouped_by_year_month_added_c_x.csv
│   ├── grouped_by_year_month_day.csv
│   └── insterted_clusters.csv
├── forecasting_models/
│   ├── daily_hourly_forecast/
│   │   ├── conv_lstm.ipynb
│   │   └── conv_lstm_site.ipynb
│   ├── monthly_forecast/
│   │   ├── arima_family.ipynb
│   │   └── xgboost.ipynb
│   └── weekly_forecast/
│       ├── arima_family.ipynb
│       └── conv_lstm.ipynb
├── EDEncAdmissions.ipynb
├── analytics.ipynb
├── deliverable.ipynb
├── requirements.txt
└── utils.py
```

## What each folder contains

### `clustering/`

Contains the clustering notebook used to explore reasons for emergency department visits. The notebook imports text vectorization, clustering, dimensionality reduction, and cluster-quality metrics, which suggests it is used to build and evaluate clusters over visit-reason text.

### `forecasting_models/`

Contains forecasting experiments organized by time scale.

* `daily_hourly_forecast/`: deep-learning forecasting notebooks for hourly or daily sequence prediction.
* `weekly_forecast/`: weekly-level forecasting experiments, including ARIMA-family and ConvLSTM approaches.
* `monthly_forecast/`: monthly forecasting experiments, including ARIMA-family and XGBoost approaches.

### `data/`

Contains the source dataset and several derived CSVs used in analysis and modeling.

* `DSU-Dataset.csv`: the main raw dataset.
* `forecast.csv` and `forecast_sep_oct.csv`: forecast-related outputs.
* `grouped_by_hour_blocks.csv`, `grouped_by_site_hour_block.csv`, `grouped_by_year_month.csv`, `grouped_by_year_month_added_c_x.csv`, `grouped_by_year_month_day.csv`: aggregated tables used for modeling and analysis.
* `insterted_clusters.csv`: clustered output used downstream in analysis.

## Core notebooks

### `analytics.ipynb`

Main analysis notebook. It loads the shared utility functions, checks the utility version, sets plotting defaults, and performs exploratory analysis over the encounter dataset.

### `clustering/clustering.ipynb`

Clustering notebook for visit-reason analysis. It cleans text, computes unique reasons, summarizes counts, and works with vectorization and clustering methods.

### `EDEncAdmissions.ipynb`

Notebook focused on ED encounters and admissions. It loads the raw dataset, converts dates, removes the reason text column, groups encounters by site, date, and hour, and summarizes hourly encounter and admission counts.

### `deliverable.ipynb`

Large notebook that appears to contain the final project output and model results, including forecasting summaries and statistical diagnostics.

## Shared utilities

### `utils.py`

Shared helper module for vectorization, clustering, elbow detection, cluster remapping, and time-series diagnostics.

Notable functions include:

* `vectorize(...)`: builds TF-IDF or BioWordVec-style embeddings.
* `kmeans_model(...)` and `bkmeans_model(...)`: run weighted or unweighted clustering.
* `find_elbow(...)`: compares inertia and silhouette scores across cluster counts.
* `_remap(...)`: maps a reason to its predicted cluster.
* `mean_over_time(...)` and `var_over_time(...)`: compute running statistics for stationarity checks.
* `sanity_check()`: returns a version string used to confirm the utility module is loaded.

## Requirements

Install the packages listed in `requirements.txt`.

Main dependencies include:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* kneed
* scipy
* fasttext
* statsmodels
* statsforecast
* utilsforecast
* xgboost
* tensorflow

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the notebooks in the order that matches your workflow:

1. Start with `analytics.ipynb` or `clustering/clustering.ipynb` for exploratory analysis.
2. Use `EDEncAdmissions.ipynb` for encounter and admissions aggregation.
3. Open the forecasting notebooks under `forecasting_models/` to reproduce the time-series experiments.
4. Review `deliverable.ipynb` for the final modeling results and reporting output.
