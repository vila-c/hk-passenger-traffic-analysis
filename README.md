# Hong Kong Cross-Border Passenger Traffic Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Framework](https://img.shields.io/badge/Framework-CRISP--DM-009688)]()
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Data](https://img.shields.io/badge/Records-56%2C424+-green)]()
[![Dashboard](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit)](https://hk-passenger-traffic-analysis.streamlit.app/)

A comprehensive data science project analysing daily cross-border passenger traffic at Hong Kong's 17 border control points from 2021 to 2025, following the CRISP-DM methodology. The analysis covers the full post-COVID recovery trajectory -- from near-zero crossings during border closures through the reopening on 8 January 2023 to present-day travel patterns.

**Author:** Vila Chung, BASc Social Data Science, The University of Hong Kong, 2025

---

## Key Findings

- **Post-reopening recovery is strong.** Year-over-year growth is significant, with the `Year` feature contributing +151,639 daily passengers per year in regression.
- **Weekends dominate traffic.** Weekend days average +87,662 more passengers than weekdays; the rule `{Weekend, Year2025} -> {VeryHighTraffic}` holds with 95.2% confidence.
- **Dual holidays amplify surges.** When both Hong Kong and Mainland holidays overlap (`Is_Both_Holiday`), traffic peaks are markedly higher than single-region holidays alone.
- **Winter 2023 was still in recovery.** The association rule `{Winter, Year2023} -> {LowTraffic}` (conf=0.699) captures the early reopening period before travel fully normalised.
- **Decision Tree outperforms Logistic Regression** for classifying high-traffic days (89.9% accuracy, AUC 0.926 vs. 82.1% accuracy, AUC 0.925).
- **Four distinct traffic regimes exist:** Holiday Peak, Weekend Peak, Regular Weekday, and Early Recovery clusters identified via K-Means (k=4, Silhouette=0.2845).

---

## Repository Structure

### Branch Overview

| Branch | Purpose |
|--------|---------|
| `main` | Stable production branch — app.py, README, requirements.txt, processed data |
| `dev` | Integration branch — merges feature branches before promoting to main |
| `raw` | Raw data ingestion only |
| `data` | Data pipeline and processed CSV |
| `feature/data-cleaning` | NB01: Data cleaning and feature engineering |
| `feature/eda-statistics` | NB02: Exploratory data analysis and statistical tests |
| `feature/classification` | NB03: Decision Tree & Logistic Regression |
| `feature/regression` | NB04: Multiple Linear Regression |
| `feature/clustering-arm` | NB05: K-Means Clustering & Association Rule Mining |
| `feature/app-dashboard` | Streamlit dashboard (app.py) |

### File Structure (main branch)

```
hk-passenger-traffic-analysis/
|
|-- 01_Data_Cleaning_and_Preparation.ipynb   # NB01: Data ingestion, cleaning, feature engineering
|-- 02_EDA_and_Statistics.ipynb              # NB02: Exploratory data analysis, statistical tests
|-- 03_Classification_Models.ipynb           # NB03: Decision Tree & Logistic Regression
|-- 04_Regression.ipynb                      # NB04: Multiple Linear Regression
|-- 05_Clustering_ARM.ipynb                  # NB05: K-Means Clustering & Association Rule Mining
|
|-- app.py                                   # Streamlit interactive dashboard
|-- requirements.txt                         # Python dependencies
|
|-- statistics_on_daily_passenger_traffic.csv # Raw data from data.gov.hk
|-- daily_traffic_processed.csv              # Cleaned and feature-engineered dataset
|
|-- fig_01_timeseries_full.png               # Full time-series plot
|-- fig_02_dow_avg.png                       # Day-of-week average traffic
|-- fig_03_*.png                             # Classification model figures
|-- fig_04_*.png                             # Regression model figures
|-- fig_05_*.png                             # Clustering & ARM figures
|-- fig_06_correlation_heatmap.png           # Feature correlation heatmap
|-- fig_07_weekend_vs_weekday_test.png       # Statistical test visualisation
|-- fig_08_yoy_growth.png                    # Year-over-year growth
|
|-- README.md                                # This file
```

---

## Methodology (CRISP-DM)

This project follows the six phases of the **Cross-Industry Standard Process for Data Mining**:

| Phase | Notebook | Description |
|-------|----------|-------------|
| **1. Business Understanding** | -- | Define objectives: understand cross-border traffic patterns, identify drivers of high-traffic days, and support policy/planning decisions. |
| **2. Data Understanding** | NB01, NB02 | Ingest 56,424+ records from the Immigration Department. Profile 17 control points, check distributions, and identify data quality issues. |
| **3. Data Preparation** | NB01 | Clean records, aggregate to daily totals, engineer temporal and holiday features. Output: 1,156-row post-reopening dataset with 21 columns. |
| **4. Modelling** | NB03, NB04, NB05 | Train classification (Decision Tree, Logistic Regression), regression (Multiple Linear Regression), clustering (K-Means), and association rule mining (Apriori). |
| **5. Evaluation** | NB03, NB04, NB05 | Evaluate with accuracy, AUC, cross-validation, R-squared, RMSE, silhouette score, support/confidence/lift. |
| **6. Deployment** | app.py | Interactive Streamlit dashboard for stakeholder exploration. |

---

## Model Performance Summary

### Classification (NB03) -- High-Traffic Day Prediction

10 features: `Month`, `DayOfWeek`, `Quarter`, `Is_Holiday`, `Is_Both_Holiday`, `Is_Weekend`, `Is_CNY`, `Is_GoldenWeek`, `Is_Easter`, `Year`

| Model | Accuracy | AUC | CV Accuracy |
|-------|----------|-----|-------------|
| **Decision Tree** | **89.91%** | **0.9264** | 88.98% |
| Logistic Regression | 82.11% | 0.9253 | 85.68% |

### Regression (NB04) -- Daily Passenger Count

| Metric | Value |
|--------|-------|
| R-squared (test) | 0.7423 |
| RMSE | 111,145 |
| Strongest positive predictor | `Year` (+151,639 per year) |
| Second strongest | `Is_Weekend` (+87,662) |

> **Note:** Negative coefficients for `Is_CNY` and `Is_GoldenWeek` are artefacts of multicollinearity with the broader `Is_Holiday` feature, not genuine negative effects.

### Clustering (NB05) -- Traffic Regime Segmentation

| Cluster | Label | Size | Description |
|---------|-------|------|-------------|
| 0 | Holiday Peak | 540 | High-traffic holiday and festival periods |
| 1 | Weekend Peak | 276 | Elevated weekend traffic |
| 2 | Regular Weekday | 244 | Baseline weekday patterns |
| 3 | Early Recovery | 29 | Low-traffic early reopening days (Jan--Feb 2023) |

K-Means k=4, Silhouette Score = 0.2845

### Association Rule Mining (NB05)

56 rules generated (min\_support=0.05, min\_confidence=0.60)

| Rule | Confidence | Lift |
|------|-----------|------|
| {Weekend, Year2025} -> {VeryHighTraffic} | 0.952 | 4.114 |
| {Winter, Year2023} -> {LowTraffic} | 0.699 | 2.633 |

---

## Holiday Feature Engineering

A key contribution of this project is the dual-region holiday feature set, capturing the unique dynamics of Hong Kong--Mainland China cross-border travel.

> **Scope note:** All day counts below are totals across the **post-reopening study period** (2023-01-08 to 2026-03-08). Annual counts vary — for example, Hong Kong gazetted 17–20 public holidays per year, and Mainland China's extended festival arrangements differ annually. Refer to NB01 for the full per-year holiday lists and sources.

| Feature | Description | Total in study period |
|---------|-------------|----------------------|
| `Is_HK_Holiday` | Hong Kong gazetted public holidays and their observed days | 52 days |
| `Is_ML_Holiday` | Mainland China public holidays (including extended Golden Week, Spring Festival) | 82 days |
| `Is_Both_Holiday` | Days that are holidays in **both** regions simultaneously | 29 days |
| `Is_Any_Holiday` / `Is_Holiday` | Union of HK and ML holidays (either or both). `Is_Holiday` is an alias for `Is_Any_Holiday` | 105 days |

**Festival-specific flags:**

| Feature | Description | Total in study period |
|---------|-------------|----------------------|
| `Is_CNY` | Chinese New Year / Spring Festival period (HK gazetted days) | 11 days |
| `Is_GoldenWeek` | Mainland Golden Week (National Day, 1--7 October) | 21 days |
| `Is_Easter` | Easter holiday period (Good Friday through Easter Monday) | 12 days |

The `Is_Both_Holiday` feature captures dual-holiday surges where travellers from both sides of the border are on holiday simultaneously, producing the highest traffic volumes in the dataset.

---

## Dataset

- **Raw records:** 56,424+ daily passenger traffic entries across 17 control points
- **Post-reopening dataset:** 1,156 rows (2023-01-08 to 2026-03-08), 21 columns
- **Source:** [Hong Kong Immigration Department via data.gov.hk](https://data.gov.hk/en-data/dataset/hk-immd-set4-statistics-daily-passenger-traffic)

The 17 control points include airports, land crossings (Lo Wu, Lok Ma Chau), sea ports, and the Hong Kong-Zhuhai-Macao Bridge.

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/vila-c/hk-passenger-traffic-analysis.git
cd hk-passenger-traffic-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run the Notebooks

Open the notebooks in order to reproduce the full analysis pipeline:

```bash
jupyter notebook
```

1. `01_Data_Cleaning_and_Preparation.ipynb` -- data ingestion and feature engineering
2. `02_EDA_and_Statistics.ipynb` -- exploratory analysis and hypothesis tests
3. `03_Classification_Models.ipynb` -- train and evaluate classifiers
4. `04_Regression.ipynb` -- regression modelling
5. `05_Clustering_ARM.ipynb` -- clustering and association rules

### Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Streamlit Dashboard

**[▶ Launch Live Dashboard](https://hk-passenger-traffic-analysis.streamlit.app/)**

The interactive dashboard (`app.py`) provides stakeholders with a visual interface to explore the analysis results without running notebooks. Features include:

- **Time-series exploration** -- filter by date range and control point
- **Day-of-week and seasonal patterns** -- interactive charts of traffic by weekday, month, and year
- **Model results viewer** -- classification metrics, regression coefficients, and cluster visualisations
- **Holiday impact analysis** -- compare traffic on regular days, single-region holidays, and dual holidays

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn |
| Association Rules | efficient-apriori |
| Statistical Tests | SciPy |
| Dashboard | Streamlit |
| Notebooks | Jupyter Notebook |
| Methodology | CRISP-DM |

---

## Data Source

**Traffic dataset:** Statistics on Daily Passenger Traffic, published by the **Hong Kong Immigration Department** and distributed via [data.gov.hk](https://data.gov.hk/en-data/dataset/hk-immd-set4-statistics-daily-passenger-traffic). Provided under the [Hong Kong Government Open Data Terms of Use](https://data.gov.hk/en/terms-and-conditions).

**Mainland China public holiday schedules:** Official announcements published by the **State Council of the People's Republic of China** via [english.www.gov.cn](https://english.www.gov.cn/). Holiday dates for 2023–2026 were manually compiled from annual government notices.

---

## Public Holiday Sources

| Region | Years | Source |
|--------|-------|--------|
| Hong Kong | 2023 | [gov.hk/holiday/2023](https://www.gov.hk/en/about/abouthk/holiday/2023.htm) |
| Hong Kong | 2024 | [gov.hk/holiday/2024](https://www.gov.hk/en/about/abouthk/holiday/2024.htm) |
| Hong Kong | 2025 | [gov.hk/holiday/2025](https://www.gov.hk/en/about/abouthk/holiday/2025.htm) |
| Hong Kong | 2026 | [gov.hk/holiday/2026](https://www.gov.hk/en/about/abouthk/holiday/2026.htm) |
| Mainland China | 2023 | [english.www.gov.cn](https://english.www.gov.cn/) |
| Mainland China | 2024 | [english.www.gov.cn](https://english.www.gov.cn/) |
| Mainland China | 2025 | [english.www.gov.cn](https://english.www.gov.cn/) |
| Mainland China | 2026 | [english.www.gov.cn](https://english.www.gov.cn/) |

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See [LICENSE](LICENSE) for details.

---

## Author

**Vila Chung** · BASc Social Data Science · The University of Hong Kong · 2025
[GitHub](https://github.com/vila-c)
