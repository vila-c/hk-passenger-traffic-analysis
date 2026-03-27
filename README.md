# 🚂 Hong Kong Cross-Border Passenger Traffic Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end data science project analysing **56,424 daily passenger traffic records** across Hong Kong's 17 border control points to uncover post-COVID recovery patterns, build predictive models, and surface actionable travel demand insights. Holiday features incorporate **both Hong Kong and Mainland China public holidays** (2023–2026) to capture cross-border demand drivers from both sides.

## 🌐 Live Demo
👉 [View Interactive Dashboard](#) *(coming soon)*

> Features an interactive **Traffic Explorer** — filter by control point, date range, and visitor type to visualise cross-border flow trends and model predictions.

---

## 🔄 CRISP-DM Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

| Phase | Notebook / File | Description |
|-------|----------------|-------------|
| Business Understanding | — | Analyse post-COVID cross-border recovery and identify traffic demand drivers |
| Data Understanding | `01_Cleaning` · `02_EDA` | Initial exploration, statistical testing, seasonal decomposition |
| Data Preparation | `01_Cleaning` | Cleaning, holiday labelling, festival flags, feature engineering |
| Modelling | `03_Classification` · `04_Regression` · `05_Clustering_ARM` | Decision Tree, Logistic Regression, Linear Regression, K-Means, Apriori |
| Evaluation | `03_Classification` · `04_Regression` · `05_Clustering_ARM` | AUC, Accuracy, R², RMSE, Silhouette Score, Lift |
| Deployment | `app/dashboard.py` | Streamlit interactive dashboard |

> **Note on methodology ordering:** In this project, we intentionally run `01_Cleaning` before `02_EDA`. While CRISP-DM typically places Data Understanding before Data Preparation, our workflow reflects the iterative nature of the framework.
> - We first perform initial cleaning and basic checks in `01_Cleaning` to ensure data quality and consistency.
> - Then, we conduct extended exploratory analysis (EDA) in `02_EDA`, applying statistical tests and seasonal analysis on the cleaned dataset.
>
> This ordering avoids misleading insights from raw, noisy data and highlights the iterative feedback loop between Data Understanding and Data Preparation.

---

## 🤖 ML Pipeline

```
Raw Data (56,424 records · 17 control points · 2021–2025)
        ↓
Data Cleaning & Feature Engineering        [01_Cleaning]
  · Removed COVID closure period (2021–2022, near-zero traffic)
  · Dual holiday labelling: HK public holidays (gov.hk)
    + Mainland China public holidays (gov.cn), 2023–2026
  · Festival flags (Is_CNY, Is_GoldenWeek, Is_Easter,
    Is_DragonBoat, Is_MidAutumn, Is_NationalDay, Is_ChingMing)
  · Mainland-specific flags (Is_ML_SpringFestival,
    Is_ML_GoldenWeek, Is_ML_DragonBoat, Is_ML_MidAutumn)
  · Combined flag: Is_Any_Holiday (HK or Mainland)
  · Datetime features (DayOfWeek, Month, Year, IsWeekend)
        ↓
Exploratory Data Analysis                  [02_EDA]
  · Post-reopening trend analysis (1,096 days)
  · Control point comparison
  · Seasonal decomposition & holiday effect
        ↓
Classification Models                      [03_Classification]
  · Binary target: High / Low traffic
  · Decision Tree (max_depth=5, prepruning)
  · Logistic Regression
  · Stratified 75/25 split · 5-fold CV
        ↓
Regression Analysis                        [04_Regression]
  · Linear Regression (predict exact daily count)
  · Residual analysis & assumption checks
        ↓
Clustering & Association Rules             [05_Clustering_ARM]
  · K-Means (k=4, elbow method)
  · Apriori (min_support=0.05, min_confidence=0.60)
        ↓
Streamlit Dashboard                        [app/dashboard.py]
  · Interactive visualisations
  · Model performance explorer
  · Traffic pattern insights
```

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `notebooks/01_Data_Cleaning_and_Preparation.ipynb` | Data wrangling, HK + Mainland dual holiday labelling, festival flags, feature engineering |
| `notebooks/02_EDA_and_Statistics.ipynb` | EDA, statistical testing, seasonal decomposition, control point analysis |
| `notebooks/03_Classification_Models.ipynb` | Decision Tree & Logistic Regression, AUC, confusion matrix, CV |
| `notebooks/04_Regression_Analysis.ipynb` | Linear Regression, R², RMSE, residual diagnostics |
| `notebooks/05_Clustering_and_ARM.ipynb` | K-Means clustering, Apriori association rules, lift analysis |
| `src/data_cleaning.py` | Reusable cleaning and feature engineering functions |
| `src/eda.py` | EDA plotting and statistical testing utilities |
| `src/classification.py` | Classification model training and evaluation |
| `src/regression.py` | Regression model training and evaluation |
| `src/clustering_arm.py` | Clustering and association rule mining |
| `app/dashboard.py` | Streamlit interactive dashboard |

---

## 🔍 Key Findings

- **Easter produces the highest average daily traffic: 1,014,569** — surpassing all other day types including CNY and Golden Week
- **Regular weekday average: 693,475** — baseline demand showing strong post-COVID recovery
- **Decision Tree identifies weekend + year as top splits** — temporal features dominate traffic prediction
- **K-Means reveals 4 distinct traffic regimes** — Holiday Peak, Weekend Peak, Regular Weekday, and Early Recovery clusters
- **Top association rule: {Weekend, Year2025} → {VeryHighTraffic}** — confidence 0.99, lift 3.51, indicating near-certain high traffic on 2025 weekends
- **Decision Tree CV gap (84.55% vs 66.61%)** — suggests moderate overfitting; Logistic Regression generalises better (83.18% vs 76.55%)
- **Dual-holiday effect:** When HK and Mainland holidays overlap (e.g., CNY, National Day), traffic surges are amplified compared to single-region holidays

---

## 📊 Model Performance

### Classification

| Metric | Decision Tree | Logistic Regression |
|--------|:------------:|:-------------------:|
| Max Depth / Regularisation | 5 (prepruning) | Default (L2) |
| Test Accuracy | 84.55% | 83.18% |
| Test AUC | 0.9296 | 0.9097 |
| 5-fold CV Accuracy | 66.61% | 76.55% |

> **Note on generalisation:** Decision Tree achieves higher test accuracy but a wider CV gap (17.94 pp), indicating overfitting to the test split. Logistic Regression shows a smaller CV gap (6.63 pp) and generalises more reliably to unseen data.

### Regression

| Metric | Linear Regression |
|--------|:-----------------:|
| R² | 0.6868 |
| RMSE | 138,736 |

### Clustering & Association Rules

| Method | Key Result |
|--------|------------|
| K-Means (k=4) | Holiday Peak · Regular Weekday · Early Recovery · Weekend Peak |
| Apriori Top Rule | {Weekend, Year2025} → {VeryHighTraffic} · conf=0.99 · lift=3.51 |
| Apriori Settings | min_support=0.05 · min_confidence=0.60 |

---

## ⚠️ Tool Suitability Notes

This project is built for **educational and portfolio purposes**. Some tools used here would require different choices in a production environment:

| Tool | Used For | Production Consideration |
|------|----------|--------------------------|
| Streamlit | Interactive dashboard | ✅ Rapid prototyping · ⚠️ Not designed for enterprise-scale traffic |
| scikit-learn | Classification & Regression | ✅ Education · ⚠️ Production pipelines typically use MLflow + model registry |
| mlxtend | Apriori association rules | ✅ Small-medium datasets · ⚠️ Use Spark FP-Growth for large-scale mining |
| pandas (in-memory) | Data processing | ✅ Up to ~1M rows · ⚠️ Use Spark / Dask for large-scale data |
| matplotlib / seaborn | Static visualisations | ✅ Publication-quality plots · ⚠️ Use Plotly / D3.js for interactive production dashboards |

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Plotly · Streamlit · mlxtend · SciPy

---

## 📊 Data Source

[Statistics on Daily Passenger Traffic](https://data.gov.hk/en-data/dataset/hk-immd-set4-statistics-daily-passenger-traffic) · Hong Kong Immigration Department · data.gov.hk · For educational purposes only.

> This is an official government open dataset covering daily passenger traffic at all Hong Kong boundary control points. The dataset spans 2021–2025 and includes breakdowns by control point, direction (Arrival/Departure), and traveller type (HK Residents, Mainland Visitors, Other Visitors).

**Public Holiday Sources:**

| Region | Years | Source |
|--------|-------|--------|
| Hong Kong | 2023 | [gov.hk/holiday/2023](https://www.gov.hk/en/about/abouthk/holiday/2023.htm) |
| Hong Kong | 2024 | [gov.hk/holiday/2024](https://www.gov.hk/en/about/abouthk/holiday/2024.htm) |
| Hong Kong | 2025 | [gov.hk/holiday/2025](https://www.gov.hk/en/about/abouthk/holiday/2025.htm) |
| Hong Kong | 2026 | [gov.hk/holiday/2026](https://www.gov.hk/en/about/abouthk/holiday/2026.htm) |
| Mainland China | 2023 | [english.www.gov.cn](https://english.www.gov.cn/policies/latestreleases/202212/09/content_WS63920eafc6d0a757729e422c.html) |
| Mainland China | 2024 | [english.www.gov.cn](https://english.www.gov.cn/policies/latestreleases/202310/25/content_WS65387be8c6d0868f4e8e0a04.html) |
| Mainland China | 2025 | [english.www.gov.cn](https://english.www.gov.cn/policies/latestreleases/202411/12/content_WS67331db5c6d0868f4e8ecd92.html) |
| Mainland China | 2026 | [english.www.gov.cn](https://english.www.gov.cn/policies/featured/202511/04/content_WS6909c915c6d00ca5f9a074f7.html) |

---

## 👤 Author

Vila Chung · HKU BASc Social Data Science · 2025
[GitHub](https://github.com/vila-c/hk-passenger-traffic-analysis)
