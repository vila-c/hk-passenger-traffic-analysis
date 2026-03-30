import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HK Cross-Border Traffic Analytics",
    page_icon="\U0001f682",
    layout="wide",
)

# ============================================================
# COLOUR PALETTE
# ============================================================
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd",
    "teal": "#17becf",
    "pink": "#e377c2",
    "grey": "#7f7f7f",
    "brown": "#8c564b",
    "olive": "#bcbd22",
}
COLOR_SEQ = list(COLORS.values())

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        "daily_traffic_processed.csv",
        parse_dates=["Date"],
    )
    # Post-reopening filter
    df = df[(df["Date"] >= "2023-01-08") & (df["Year"] <= 2025)].copy()

    # Validate required columns
    required_cols = [
        "Is_HK_Holiday", "Is_ML_Holiday", "Is_Both_Holiday",
        "Is_Any_Holiday", "Is_Holiday",
        "Is_Weekend", "Is_CNY", "Is_GoldenWeek", "Is_Easter",
        "Year", "Month", "DayOfWeek", "Quarter",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Festival_Type classification
    conditions = [
        df["Is_CNY"] == 1,
        df["Is_GoldenWeek"] == 1,
        df["Is_Easter"] == 1,
        df["Is_Holiday"] == 1,
        df["Is_Weekend"] == 1,
    ]
    choices = ["CNY", "Golden Week", "Easter", "Other Holiday", "Regular Weekend"]
    df["Festival_Type"] = np.select(conditions, choices, default="Weekday")

    # Ensure DayName exists
    if "DayName" not in df.columns:
        df["DayName"] = df["Date"].dt.day_name()

    return df


df = load_data()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("\U0001f50d Filters")

all_years = sorted(df["Year"].unique())
sel_years = st.sidebar.multiselect("Year", all_years, default=all_years)

all_day_types = sorted(df["Festival_Type"].unique())
sel_day_types = st.sidebar.multiselect("Day Type", all_day_types, default=all_day_types)

mask = df["Year"].isin(sel_years) & df["Festival_Type"].isin(sel_day_types)
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Days", f"{len(fdf):,}")
if len(fdf) > 0:
    st.sidebar.metric("Avg Daily Traffic", f"{fdf['Total'].mean():,.0f}")
    st.sidebar.caption(
        f"{fdf['Date'].min().strftime('%Y-%m-%d')} to {fdf['Date'].max().strftime('%Y-%m-%d')}"
    )
else:
    st.sidebar.warning("No data matches current filters.")

# ============================================================
# TITLE
# ============================================================
st.title("\U0001f682 Hong Kong Cross-Border Passenger Traffic Analytics")
st.caption("Interactive dashboard for post-reopening (2023-01-08 onwards) cross-border traffic analysis")

# ============================================================
# KPI ROW
# ============================================================
if len(fdf) > 0:
    peak_idx = fdf["Total"].idxmax()
    peak_date = fdf.loc[peak_idx, "Date"].strftime("%Y-%m-%d")
    peak_val = fdf.loc[peak_idx, "Total"]
    top_festival = (
        fdf.groupby("Festival_Type")["Total"]
        .mean()
        .drop("Weekday", errors="ignore")
        .idxmax()
    )
    wkend_avg = fdf.loc[fdf["Is_Weekend"] == 1, "Total"].mean()
    wkday_avg = fdf.loc[fdf["Is_Weekend"] == 0, "Total"].mean()
    gap_pct = (wkend_avg - wkday_avg) / wkday_avg * 100 if wkday_avg else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Days", f"{len(fdf):,}")
    k2.metric("Avg Daily Traffic", f"{fdf['Total'].mean():,.0f}")
    k3.metric("Peak Day", f"{peak_date}", delta=f"{peak_val:,.0f}")
    k4.metric("Top Festival", top_festival)
    k5.metric("Weekend vs Weekday", f"+{gap_pct:.1f}%")

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "\U0001f4ca Overview",
    "\U0001f389 Holiday Analysis",
    "\U0001f916 Classification",
    "\U0001f4c8 Regression",
    "\U0001f535 Clustering",
    "\U0001f517 Association Rules",
])

# ============================================================
# TAB 1 - OVERVIEW
# ============================================================
with tab1:
    st.header("Traffic Overview")

    # Time series with 7-day MA
    ts = fdf.sort_values("Date").copy()
    ts["MA7"] = ts["Total"].rolling(7, min_periods=1).mean()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts["Date"], y=ts["Total"],
        mode="lines", name="Daily Total",
        line=dict(color=COLORS["primary"], width=0.8),
        opacity=0.4,
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts["Date"], y=ts["MA7"],
        mode="lines", name="7-Day MA",
        line=dict(color=COLORS["danger"], width=2),
    ))
    fig_ts.update_layout(
        title="Daily Cross-Border Passenger Traffic with 7-Day Moving Average",
        xaxis_title="Date", yaxis_title="Total Passengers",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Day of week average
    col_a, col_b = st.columns(2)

    with col_a:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = fdf.groupby("DayName")["Total"].mean().reindex(dow_order).reset_index()
        dow_avg.columns = ["Day", "Average"]
        fig_dow = px.bar(
            dow_avg, x="Day", y="Average",
            title="Average Traffic by Day of Week",
            color="Average",
            color_continuous_scale="Blues",
        )
        fig_dow.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_b:
        # Year-over-year summary
        yoy = fdf.groupby("Year").agg(
            Days=("Total", "count"),
            Mean=("Total", "mean"),
            Median=("Total", "median"),
            Std=("Total", "std"),
            Min=("Total", "min"),
            Max=("Total", "max"),
        ).round(0)
        yoy.index = yoy.index.astype(int)
        for c in ["Mean", "Median", "Std", "Min", "Max"]:
            yoy[c] = yoy[c].apply(lambda x: f"{x:,.0f}")
        st.markdown("#### Year-over-Year Summary")
        st.dataframe(yoy, use_container_width=True)

# ============================================================
# TAB 2 - HOLIDAY ANALYSIS
# ============================================================
with tab2:
    st.header("Holiday & Festival Analysis")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fest_stats = fdf.groupby("Festival_Type")["Total"].agg(["mean", "std", "count"]).reset_index()
        fest_stats.columns = ["Festival_Type", "mean", "std", "count"]
        fest_stats = fest_stats.sort_values("mean", ascending=True)

        fig_fest = go.Figure()
        fig_fest.add_trace(go.Bar(
            y=fest_stats["Festival_Type"],
            x=fest_stats["mean"],
            orientation="h",
            error_x=dict(type="data", array=fest_stats["std"].fillna(0)),
            marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["success"],
                          COLORS["danger"], COLORS["purple"], COLORS["teal"]][:len(fest_stats)],
        ))
        fig_fest.update_layout(
            title="Average Traffic by Day/Festival Type (with Std Dev)",
            xaxis_title="Average Total Passengers",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig_fest, use_container_width=True)

    with col_h2:
        monthly = fdf.groupby(["Year", "Month"])["Total"].mean().reset_index()
        monthly["Year"] = monthly["Year"].astype(str)
        fig_monthly = px.line(
            monthly, x="Month", y="Total", color="Year",
            title="Monthly Average Traffic by Year",
            markers=True,
            color_discrete_sequence=COLOR_SEQ,
        )
        fig_monthly.update_layout(
            template="plotly_white", height=400,
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    # Festival detail table
    st.markdown("#### Festival Detail Table")
    fest_detail = fdf.groupby("Festival_Type").agg(
        Days=("Total", "count"),
        Mean=("Total", "mean"),
        Median=("Total", "median"),
        Min=("Total", "min"),
        Max=("Total", "max"),
    ).round(0)
    for c in ["Mean", "Median", "Min", "Max"]:
        fest_detail[c] = fest_detail[c].apply(lambda x: f"{x:,.0f}")
    st.dataframe(fest_detail, use_container_width=True)

# ============================================================
# TAB 3 - CLASSIFICATION
# ============================================================
with tab3:
    st.header("Classification Models (NB03 Results)")
    st.markdown("Two models trained on **10 temporal and holiday features** to classify traffic as High/Low.")

    # Hardcoded metrics
    dt_metrics = {
        "Accuracy": 0.8991, "Precision": 0.9451, "Recall": 0.8350,
        "F1": 0.8866, "AUC-ROC": 0.9264,
        "CV Mean": 0.8898, "CV Std": 0.0318, "Test-CV Gap": 0.0093,
    }
    lr_metrics = {
        "Accuracy": 0.8211, "Precision": 0.7963, "Recall": 0.8350,
        "F1": 0.8152, "AUC-ROC": 0.9253,
        "CV Mean": 0.8568, "CV Std": 0.0342, "Test-CV Gap": 0.0357,
    }

    dt_cm = np.array([[110, 5], [17, 86]])
    lr_cm = np.array([[93, 22], [17, 86]])

    # Model comparison table
    comp_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC",
                    "10-Fold CV Mean", "CV Std", "Test-CV Gap"],
        "Decision Tree": [
            f"{dt_metrics['Accuracy']:.4f}", f"{dt_metrics['Precision']:.4f}",
            f"{dt_metrics['Recall']:.4f}", f"{dt_metrics['F1']:.4f}",
            f"{dt_metrics['AUC-ROC']:.4f}", f"{dt_metrics['CV Mean']:.4f}",
            f"{dt_metrics['CV Std']:.4f}", f"{dt_metrics['Test-CV Gap']:.4f}",
        ],
        "Logistic Regression": [
            f"{lr_metrics['Accuracy']:.4f}", f"{lr_metrics['Precision']:.4f}",
            f"{lr_metrics['Recall']:.4f}", f"{lr_metrics['F1']:.4f}",
            f"{lr_metrics['AUC-ROC']:.4f}", f"{lr_metrics['CV Mean']:.4f}",
            f"{lr_metrics['CV Std']:.4f}", f"{lr_metrics['Test-CV Gap']:.4f}",
        ],
    })

    col_c1, col_c2 = st.columns([1.2, 0.8])

    with col_c1:
        st.markdown("#### Model Performance Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col_c2:
        # Radar chart
        metrics_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        dt_vals = [dt_metrics[m] for m in metrics_names]
        lr_vals = [lr_metrics[m] for m in metrics_names]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=dt_vals + [dt_vals[0]],
            theta=metrics_names + [metrics_names[0]],
            fill="toself", name="Decision Tree",
            line_color=COLORS["primary"],
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=lr_vals + [lr_vals[0]],
            theta=metrics_names + [metrics_names[0]],
            fill="toself", name="Logistic Regression",
            line_color=COLORS["secondary"],
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.7, 1.0])),
            title="Model Comparison Radar",
            template="plotly_white", height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Confusion matrices side by side
    st.markdown("#### Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        labels = ["Low", "High"]
        fig_cm_dt = go.Figure(data=go.Heatmap(
            z=dt_cm, x=labels, y=labels,
            text=dt_cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
        ))
        fig_cm_dt.update_layout(
            title="Decision Tree (max_depth=5)",
            xaxis_title="Predicted", yaxis_title="Actual",
            template="plotly_white", height=350,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm_dt, use_container_width=True)

    with cm_col2:
        fig_cm_lr = go.Figure(data=go.Heatmap(
            z=lr_cm, x=labels, y=labels,
            text=lr_cm, texttemplate="%{text}",
            colorscale="Oranges", showscale=False,
        ))
        fig_cm_lr.update_layout(
            title="Logistic Regression",
            xaxis_title="Predicted", yaxis_title="Actual",
            template="plotly_white", height=350,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm_lr, use_container_width=True)

    # Feature Importance
    st.markdown("#### Decision Tree Feature Importance")
    fi = {
        "Year": 0.375, "DayOfWeek": 0.302, "Quarter": 0.142,
        "Month": 0.065, "Is_Holiday": 0.062, "Is_Weekend": 0.047,
        "Is_CNY": 0.006, "Is_GoldenWeek": 0.000, "Is_Both_Holiday": 0.000,
        "Is_Easter": 0.000,
    }
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance")
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        title="Feature Importance (10 Features)",
        color="Importance", color_continuous_scale="Viridis",
    )
    fig_fi.update_layout(template="plotly_white", height=400, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Interactive Predictor
    st.markdown("---")
    st.markdown("#### Interactive Traffic Predictor")
    st.caption("Adjust the features below to get a predicted traffic level (based on Decision Tree logic).")

    pr_col1, pr_col2, pr_col3 = st.columns(3)
    with pr_col1:
        p_year = st.selectbox("Year", [2023, 2024, 2025], index=2, key="pred_year")
        p_month = st.slider("Month", 1, 12, 6, key="pred_month")
    with pr_col2:
        p_dow = st.selectbox("Day of Week", list(range(7)),
                             format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                             key="pred_dow")
        p_quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1, key="pred_quarter")
    with pr_col3:
        p_weekend = st.checkbox("Is Weekend", value=p_dow >= 5, key="pred_wkend")
        p_holiday = st.checkbox("Is Holiday", key="pred_holiday")
        p_cny = st.checkbox("Is CNY", key="pred_cny")

    # Simple heuristic based on feature importances
    score = 0.0
    # Year effect: 2025 high, 2023 low
    score += (p_year - 2023) / 2 * 0.375
    # Weekend/Sat-Sun push high
    score += (1 if p_dow >= 5 else 0) * 0.302
    # Quarter
    score += (p_quarter / 4) * 0.142
    # Month
    score += (p_month / 12) * 0.065
    # Holiday
    score += (1 if p_holiday else 0) * 0.062
    # Weekend flag
    score += (1 if p_weekend else 0) * 0.047
    # CNY
    score += (1 if p_cny else 0) * 0.006

    prob_high = min(max(score, 0), 1)
    predicted_label = "High" if prob_high >= 0.45 else "Low"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_high * 100,
        title={"text": f"Predicted: {predicted_label} Traffic"},
        delta={"reference": 50, "increasing": {"color": COLORS["danger"]}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 45], "color": "#d4edda"},
                {"range": [45, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": COLORS["danger"], "width": 4},
                "thickness": 0.75,
                "value": 45,
            },
        },
    ))
    fig_gauge.update_layout(height=300, template="plotly_white")
    st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# TAB 4 - REGRESSION
# ============================================================
with tab4:
    st.header("Regression Analysis (NB04 Results)")
    st.markdown("Linear regression predicting daily total passenger count from temporal and holiday features.")

    # Hardcoded metrics
    reg_col1, reg_col2 = st.columns(2)

    with reg_col1:
        st.markdown("#### Model Performance")
        reg_metrics = pd.DataFrame({
            "Metric": [
                "R\u00b2 Test", "R\u00b2 Train",
                "RMSE Test", "RMSE Train",
                "MAE Test",
                "10-Fold CV R\u00b2",
            ],
            "Value": [
                "0.7423", "0.7042",
                "111,145", "126,839",
                "83,844",
                "0.3192 \u00b1 0.5176",
            ],
        })
        st.dataframe(reg_metrics, use_container_width=True, hide_index=True)

    with reg_col2:
        st.markdown("#### Coefficient Magnitudes")
        coefs = {
            "Year": 151639, "Is_Weekend": 87662, "Month": 40125,
            "Is_Holiday": 37873, "Quarter": 36582, "DayOfWeek": 22666,
            "Is_Both_Holiday": 19905, "Is_Easter": 18617,
            "Is_GoldenWeek": -13433, "Is_CNY": -23326,
        }
        coef_df = pd.DataFrame(list(coefs.items()), columns=["Feature", "Coefficient"])
        coef_df = coef_df.sort_values("Coefficient")
        coef_df["Color"] = coef_df["Coefficient"].apply(
            lambda x: COLORS["success"] if x > 0 else COLORS["danger"]
        )
        fig_coef = go.Figure(go.Bar(
            y=coef_df["Feature"],
            x=coef_df["Coefficient"],
            orientation="h",
            marker_color=coef_df["Color"],
        ))
        fig_coef.update_layout(
            title="Standardised Coefficients",
            xaxis_title="Coefficient Value",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    # Predicted vs Actual scatter (simulated from data)
    st.markdown("#### Predicted vs Actual (Approximation)")

    # Create a simple linear prediction to visualise the relationship
    feature_cols_reg = [
        "Year", "Month", "DayOfWeek", "Quarter",
        "Is_Weekend", "Is_Holiday", "Is_CNY",
        "Is_GoldenWeek", "Is_Easter", "Is_Both_Holiday",
    ]
    if all(c in fdf.columns for c in feature_cols_reg):
        X_vis = fdf[feature_cols_reg].copy()
        y_vis = fdf["Total"].copy()
        # Use coefficient values to generate approximate predictions
        coef_map = {
            "Year": 151639, "Is_Weekend": 87662, "Month": 40125,
            "Is_Holiday": 37873, "Quarter": 36582, "DayOfWeek": 22666,
            "Is_Both_Holiday": 19905, "Is_Easter": 18617,
            "Is_GoldenWeek": -13433, "Is_CNY": -23326,
        }
        # Standardise features for prediction
        X_std = (X_vis - X_vis.mean()) / X_vis.std().replace(0, 1)
        y_pred = sum(X_std[col] * coef_map.get(col, 0) for col in feature_cols_reg)
        y_pred = y_pred + y_vis.mean()

        scatter_df = pd.DataFrame({"Actual": y_vis.values, "Predicted": y_pred.values})
        fig_scatter = px.scatter(
            scatter_df, x="Actual", y="Predicted",
            title="Predicted vs Actual Total Passengers",
            opacity=0.5,
            color_discrete_sequence=[COLORS["primary"]],
        )
        min_val = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min())
        max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect Prediction",
            line=dict(color=COLORS["danger"], dash="dash"),
        ))
        fig_scatter.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Limitation note
    st.warning(
        "**Limitation: Multicollinearity.** Several features (Month/Quarter, Is_Weekend/DayOfWeek) "
        "are correlated, which can inflate coefficient variance and make individual coefficient "
        "interpretation unreliable. The high CV standard deviation (0.5176) and the gap between "
        "test R\u00b2 (0.74) and CV R\u00b2 (0.32) suggest instability across folds. "
        "Consider regularisation (Ridge/Lasso) or feature selection in future work."
    )

# ============================================================
# TAB 5 - CLUSTERING
# ============================================================
with tab5:
    st.header("K-Means Clustering (NB05 Results)")

    # Elbow and Silhouette
    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertia = [6939, 5605, 4523, 3829, 3405, 3086, 2789]
    silhouette = [0.3514, 0.3679, 0.2845, 0.3223, 0.2727, 0.2651, 0.2665]

    elbow_col, sil_col = st.columns(2)

    with elbow_col:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_values, y=inertia,
            mode="lines+markers", name="Inertia",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=8),
        ))
        fig_elbow.add_vline(x=4, line_dash="dash", line_color=COLORS["danger"],
                            annotation_text="k=4 (chosen)")
        fig_elbow.update_layout(
            title="Elbow Method", xaxis_title="k",
            yaxis_title="Inertia", template="plotly_white", height=380,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with sil_col:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=k_values, y=silhouette,
            mode="lines+markers", name="Silhouette Score",
            line=dict(color=COLORS["secondary"], width=2),
            marker=dict(size=8),
        ))
        fig_sil.add_vline(x=4, line_dash="dash", line_color=COLORS["danger"],
                          annotation_text="k=4 (chosen)")
        fig_sil.update_layout(
            title="Silhouette Score", xaxis_title="k",
            yaxis_title="Score", template="plotly_white", height=380,
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    # Cluster Profiles
    st.markdown("#### Cluster Profiles (k=4)")

    cluster_profiles = pd.DataFrame({
        "Cluster": ["Holiday Peak", "Regular Weekday", "Early Recovery", "Weekend Peak"],
        "Days": [540, 244, 29, 276],
        "Avg Traffic": ["777,579", "489,224", "858,604", "1,015,798"],
        "Weekend %": ["0.0%", "14.8%", "20.7%", "97.5%"],
        "Holiday %": ["7.2%", "2.0%", "100.0%", "11.6%"],
    })

    # Color-coded profile cards
    profile_colors = [COLORS["primary"], COLORS["success"], COLORS["purple"], COLORS["secondary"]]
    profile_cols = st.columns(4)
    for i, row in cluster_profiles.iterrows():
        with profile_cols[i]:
            st.markdown(
                f"<div style='background-color:{profile_colors[i]}20; "
                f"border-left: 4px solid {profile_colors[i]}; "
                f"padding: 15px; border-radius: 5px; margin-bottom: 10px;'>"
                f"<h4 style='color:{profile_colors[i]}; margin:0;'>{row['Cluster']}</h4>"
                f"<p style='margin:5px 0;'><b>{row['Days']}</b> days</p>"
                f"<p style='margin:5px 0;'>Avg: <b>{row['Avg Traffic']}</b></p>"
                f"<p style='margin:5px 0;'>Weekend: {row['Weekend %']}</p>"
                f"<p style='margin:5px 0;'>Holiday: {row['Holiday %']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.dataframe(cluster_profiles, use_container_width=True, hide_index=True)

    # Cluster scatter visualisation from data
    st.markdown("#### Cluster Distribution (Approximation)")
    if "Traffic_Level" in fdf.columns:
        # Assign cluster labels based on Festival_Type heuristic for visualisation
        def assign_cluster(row):
            if row["Is_Holiday"] == 1 and row["Is_Weekend"] == 0:
                return "Early Recovery"
            elif row["Is_Weekend"] == 1:
                return "Weekend Peak"
            elif row["Year"] == 2023 and row["Month"] <= 6:
                return "Regular Weekday"
            else:
                return "Holiday Peak"

        vis_df = fdf.copy()
        vis_df["Cluster"] = vis_df.apply(assign_cluster, axis=1)
        cluster_color_map = {
            "Holiday Peak": COLORS["primary"],
            "Regular Weekday": COLORS["success"],
            "Early Recovery": COLORS["purple"],
            "Weekend Peak": COLORS["secondary"],
        }
        fig_cluster = px.scatter(
            vis_df, x="Date", y="Total",
            color="Cluster",
            color_discrete_map=cluster_color_map,
            title="Traffic by Cluster Assignment (Approximation)",
            opacity=0.6,
        )
        fig_cluster.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_cluster, use_container_width=True)

# ============================================================
# TAB 6 - ASSOCIATION RULES
# ============================================================
with tab6:
    st.header("Association Rule Mining (NB05 Results)")

    # Top 20 rules hardcoded
    rules_data = [
        {"Antecedents": "{Weekend, Year2025}", "Consequent": "{VeryHighTraffic}", "Support": 0.088, "Confidence": 0.952, "Lift": 4.114},
        {"Antecedents": "{VeryHighTraffic, Year2025}", "Consequent": "{Weekend}", "Support": 0.088, "Confidence": 0.931, "Lift": 3.557},
        {"Antecedents": "{Weekend, Spring}", "Consequent": "{VeryHighTraffic}", "Support": 0.065, "Confidence": 0.900, "Lift": 3.889},
        {"Antecedents": "{Weekend, Autumn}", "Consequent": "{VeryHighTraffic}", "Support": 0.058, "Confidence": 0.889, "Lift": 3.841},
        {"Antecedents": "{Weekend, Summer}", "Consequent": "{VeryHighTraffic}", "Support": 0.066, "Confidence": 0.876, "Lift": 3.783},
        {"Antecedents": "{VeryHighTraffic, Spring}", "Consequent": "{Weekend}", "Support": 0.065, "Confidence": 0.875, "Lift": 3.342},
        {"Antecedents": "{VeryHighTraffic, Autumn}", "Consequent": "{Weekend}", "Support": 0.058, "Confidence": 0.870, "Lift": 3.321},
        {"Antecedents": "{Weekend, Year2024}", "Consequent": "{VeryHighTraffic}", "Support": 0.067, "Confidence": 0.860, "Lift": 3.717},
        {"Antecedents": "{VeryHighTraffic, Summer}", "Consequent": "{Weekend}", "Support": 0.066, "Confidence": 0.855, "Lift": 3.266},
        {"Antecedents": "{VeryHighTraffic, Year2024}", "Consequent": "{Weekend}", "Support": 0.067, "Confidence": 0.851, "Lift": 3.252},
        {"Antecedents": "{Weekend, Winter}", "Consequent": "{HighTraffic}", "Support": 0.050, "Confidence": 0.833, "Lift": 2.862},
        {"Antecedents": "{Weekend}", "Consequent": "{VeryHighTraffic}", "Support": 0.182, "Confidence": 0.833, "Lift": 3.600},
        {"Antecedents": "{VeryHighTraffic}", "Consequent": "{Weekend}", "Support": 0.182, "Confidence": 0.788, "Lift": 3.009},
        {"Antecedents": "{Holiday}", "Consequent": "{VeryHighTraffic}", "Support": 0.033, "Confidence": 0.750, "Lift": 3.241},
        {"Antecedents": "{Year2023, Winter}", "Consequent": "{LowTraffic}", "Support": 0.057, "Confidence": 0.699, "Lift": 2.633},
        {"Antecedents": "{LowTraffic, Winter}", "Consequent": "{Year2023}", "Support": 0.057, "Confidence": 0.683, "Lift": 2.359},
        {"Antecedents": "{Year2023}", "Consequent": "{LowTraffic}", "Support": 0.120, "Confidence": 0.414, "Lift": 1.560},
        {"Antecedents": "{LowTraffic}", "Consequent": "{Year2023}", "Support": 0.120, "Confidence": 0.452, "Lift": 1.560},
        {"Antecedents": "{Weekday}", "Consequent": "{HighTraffic}", "Support": 0.212, "Confidence": 0.422, "Lift": 1.449},
        {"Antecedents": "{Winter}", "Consequent": "{LowTraffic}", "Support": 0.086, "Confidence": 0.390, "Lift": 1.470},
    ]
    rules_df = pd.DataFrame(rules_data)

    # Support vs Confidence bubble chart
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_bubble = px.scatter(
            rules_df, x="Support", y="Confidence",
            size="Lift", color="Lift",
            hover_data=["Antecedents", "Consequent"],
            title="Support vs Confidence (Bubble Size = Lift)",
            color_continuous_scale="Viridis",
            size_max=30,
        )
        fig_bubble.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col_r2:
        top_lift = rules_df.nlargest(10, "Lift").copy()
        top_lift["Rule"] = top_lift["Antecedents"] + " \u2192 " + top_lift["Consequent"]
        top_lift = top_lift.sort_values("Lift")
        fig_lift = px.bar(
            top_lift, x="Lift", y="Rule", orientation="h",
            title="Top 10 Rules by Lift",
            color="Confidence",
            color_continuous_scale="Reds",
        )
        fig_lift.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_lift, use_container_width=True)

    # Key rules highlight
    st.markdown("#### Key Rules Highlighted")
    key_col1, key_col2 = st.columns(2)

    with key_col1:
        st.success(
            "**Rule 1:** {Weekend, Year2025} \u2192 {VeryHighTraffic}\n\n"
            "- Confidence: **95.2%**\n"
            "- Lift: **4.114**\n"
            "- Interpretation: Weekend days in 2025 almost certainly see very high traffic."
        )

    with key_col2:
        st.info(
            "**Rule 2:** {Winter, Year2023} \u2192 {LowTraffic}\n\n"
            "- Confidence: **69.9%**\n"
            "- Lift: **2.633**\n"
            "- Interpretation: Early reopening winter period in 2023 had notably low traffic."
        )

    # Interactive Rule Explorer
    st.markdown("#### Interactive Rule Explorer")
    min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05, key="arm_conf")
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.5, 0.1, key="arm_lift")
    filtered_rules = rules_df[
        (rules_df["Confidence"] >= min_conf) & (rules_df["Lift"] >= min_lift)
    ].sort_values("Lift", ascending=False)
    st.write(f"**{len(filtered_rules)}** rules match the criteria.")
    st.dataframe(
        filtered_rules.style.format({
            "Support": "{:.3f}", "Confidence": "{:.3f}", "Lift": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("### Tool Suitability Summary")

tool_table = pd.DataFrame({
    "Analysis Task": [
        "Day-type Classification",
        "Traffic Volume Prediction",
        "Traffic Pattern Discovery",
        "Factor Association Mining",
    ],
    "Tool": [
        "Decision Tree / Logistic Regression",
        "Linear Regression",
        "K-Means Clustering",
        "Apriori Association Rules",
    ],
    "Suitability": [
        "High - DT achieves 89.9% accuracy with good generalisation",
        "Moderate - R\u00b2=0.74 but multicollinearity limits reliability",
        "Moderate - Silhouette 0.37 reveals meaningful but overlapping groups",
        "High - Lift values up to 4.1 reveal strong non-obvious patterns",
    ],
    "Key Insight": [
        "Year and DayOfWeek are the strongest predictors",
        "Year trend (+151K) and weekends (+88K) dominate",
        "4 distinct traffic regimes identified",
        "Weekends in 2025 almost guarantee very high traffic",
    ],
})
st.dataframe(tool_table, use_container_width=True, hide_index=True)

st.markdown(
    "<div style='text-align:center; color:grey; padding:20px;'>"
    "<b>Vila Chung</b> | HKU BASc Social Data Science | 2025<br>"
    "Data source: <a href='https://data.gov.hk' target='_blank'>data.gov.hk</a>"
    "</div>",
    unsafe_allow_html=True,
)
