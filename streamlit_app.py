import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(page_title="World Population Analytics Dashboard 🌍", layout="wide", page_icon="🌎")

# ----------------------------------------------------
# Load and Clean Data (Cached)
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/API_SP.POP.TOTL_DS2_en_csv_v2_23043.csv", skiprows=4)
    df = df.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='Population')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Population', 'Year'], inplace=True)
    df = df[df['Year'].between(1960, 2020)]
    return df

df = load_data()

# ----------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------
st.sidebar.title("🌎 Global Dashboard Controls")

selected_year = st.sidebar.slider("Select Year", 1960, 2020, 2020)
selected_country = st.sidebar.selectbox("Select a Country", sorted(df["Country Name"].unique()))

multi_countries = st.sidebar.multiselect(
    "Select Countries to Compare (All Supported)",
    sorted(df["Country Name"].unique()),
    default=["India", "China", "United States"]
)

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 World Map",
    "📊 Top 10 Countries",
    "📈 Country Trend",
    "🤖 Model Forecast",
    "🔍 Global Country Comparison"
])

# ----------------------------------------------------
# 🌍 Tab 1 — Global Map
# ----------------------------------------------------
with tab1:
    st.subheader("Global Population Animation (1960–2020)")

    fig_map = px.choropleth(
        df,
        locations="Country Code",
        color="Population",
        hover_name="Country Name",
        animation_frame="Year",
        color_continuous_scale=px.colors.sequential.Purples,
        title="World Population Growth (1960–2020)",
        projection="natural earth"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------------------------------
# 📊 Tab 2 — Top 10 Countries by Year
# ----------------------------------------------------
with tab2:
    st.subheader(f"Top 10 Most Populous Countries — {selected_year}")

    top10_df = (
        df[df["Year"] == selected_year]
        .sort_values(by="Population", ascending=False)
        .head(10)
    )

    fig_top10 = px.bar(
        top10_df,
        x="Country Name",
        y="Population",
        color="Population",
        color_continuous_scale="Purples",
        title=f"Top 10 Countries by Population in {selected_year}",
        text_auto=".2s"
    )
    st.plotly_chart(fig_top10, use_container_width=True)

# ----------------------------------------------------
# 📈 Tab 3 — Single Country Trend
# ----------------------------------------------------
with tab3:
    st.subheader(f"{selected_country} — Population Trend (1960–2020)")

    country_df = df[df["Country Name"] == selected_country]

    fig_country_trend = px.line(
        country_df,
        x="Year",
        y="Population",
        markers=True,
        color_discrete_sequence=["purple"],
        title=f"{selected_country} — Population Trend"
    )

    st.plotly_chart(fig_country_trend, use_container_width=True)

# ----------------------------------------------------
# 🤖 Tab 4 — Forecast (Single Country)
# ----------------------------------------------------
with tab4:
    st.subheader(f"{selected_country} — Forecasting Future Population (2021–2035)")

    X = country_df[["Year"]].values
    y = country_df["Population"].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(2021, 2036)
    future_preds = model.predict(future_years.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Predicted Population": future_preds
    })

    combined_df = pd.concat([
        country_df.rename(columns={"Population": "Actual Population"}).set_index("Year"),
        forecast_df.set_index("Year")
    ], axis=1).reset_index()

    fig_forecast = px.line(
        combined_df,
        x="Year",
        y=["Actual Population", "Predicted Population"],
        labels={"value": "Population", "variable": "Type"},
        color_discrete_map={
            "Actual Population": "violet",
            "Predicted Population": "purple"
        },
        title=f"{selected_country}: Historical vs Forecasted Population"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ----------------------------------------------------
# 🔍 Tab 5 — Multi-Country Comparison (ALL 196 COUNTRIES)
# ----------------------------------------------------
with tab5:
    st.subheader("🌐 Multi-Country Population Comparison (1960–2035)")

    if not multi_countries:
        st.warning("Please select at least one country to visualize.")
    else:
        multi_df = df[df["Country Name"].isin(multi_countries)]

        # Trend Comparison (Historical)
        st.markdown("### 📈 Historical Population Trend (1960–2020)")
        fig_compare_trend = px.line(
            multi_df,
            x="Year",
            y="Population",
            color="Country Name",
            line_group="Country Name",
            hover_name="Country Name",
            title="Population Comparison Across Selected Countries (1960–2020)"
        )

        # Add Global Average line
        global_avg = df.groupby("Year")["Population"].mean().reset_index()
        fig_compare_trend.add_scatter(
            x=global_avg["Year"],
            y=global_avg["Population"],
            mode="lines",
            name="🌍 Global Average",
            line=dict(color="black", dash="dot", width=3)
        )

        st.plotly_chart(fig_compare_trend, use_container_width=True)

        # Forecast Comparison (2021–2035)
        st.markdown("### 🤖 Forecast Comparison (2021–2035)")

        forecast_combined = []
        for country in multi_countries:
            cdf = df[df["Country Name"] == country]
            X = cdf[["Year"]].values
            y = cdf["Population"].values

            model = LinearRegression()
            model.fit(X, y)

            future_years = np.arange(2021, 2036)
            preds = model.predict(future_years.reshape(-1, 1))
            fdf = pd.DataFrame({
                "Country Name": country,
                "Year": future_years,
                "Predicted Population": preds
            })
            forecast_combined.append(fdf)

        forecast_all = pd.concat(forecast_combined)

        fig_forecast_all = px.line(
            forecast_all,
            x="Year",
            y="Predicted Population",
            color="Country Name",
            title="Forecast Comparison Across Selected Countries (2021–2035)"
        )

        st.plotly_chart(fig_forecast_all, use_container_width=True)

        # Download button for comparison data
        st.download_button(
            label="📥 Download Comparison Data (CSV)",
            data=forecast_all.to_csv(index=False).encode('utf-8'),
            file_name="multi_country_forecast.csv",
            mime="text/csv"
        )
