import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="World Population Dashboard", layout="wide", page_icon="🌍")

# ---------------------------
# Load Data (Cached)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/API_SP.POP.TOTL_DS2_en_csv_v2_23043.csv", skiprows=4)
    df = df.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='Population')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Population', 'Year'], inplace=True)
    return df

df = load_data()

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("🌎 Global Controls")

selected_country = st.sidebar.selectbox("Select a Country", sorted(df["Country Name"].unique()))
selected_year = st.sidebar.slider("Select Year", int(df["Year"].min()), int(df["Year"].max()), 2020)

# Multi-country comparison selection
multi_countries = st.sidebar.multiselect(
    "Compare up to 3 countries", 
    sorted(df["Country Name"].unique()),
    default=["India", "China"]
)

if len(multi_countries) > 3:
    st.sidebar.warning("⚠️ You can select up to 3 countries for comparison only.")

country_df = df[df["Country Name"] == selected_country]

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 World Map",
    "📊 Top 10 Countries",
    "📈 Country Trend",
    "🤖 Model Forecast",
    "🔍 Multi-Country Comparison"
])

# ---------------------------
# 🌍 Tab 1: World Map
# ---------------------------
with tab1:
    st.subheader("Global Population Animation (1960–2020)")
    fig_map = px.choropleth(
        df,
        locations="Country Code",
        color="Population",
        hover_name="Country Name",
        animation_frame="Year",
        color_continuous_scale=px.colors.sequential.Purples,
        projection="natural earth",
        title="World Population Growth (1960–2020)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------
# 📊 Tab 2: Top 10 Countries
# ---------------------------
with tab2:
    st.subheader(f"Top 10 Most Populous Countries in {selected_year}")
    year_df = df[df["Year"] == selected_year].sort_values(by="Population", ascending=False).head(10)
    fig_top = px.bar(
        year_df,
        x="Country Name",
        y="Population",
        text="Population",
        color="Population",
        color_continuous_scale="Purples",
        title=f"Top 10 Countries by Population — {selected_year}"
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ---------------------------
# 📈 Tab 3: Country Trend
# ---------------------------
with tab3:
    st.subheader(f"{selected_country} — Population Trend (1960–2020)")
    fig_trend = px.line(
        country_df,
        x="Year",
        y="Population",
        markers=True,
        color_discrete_sequence=["violet"],
        title=f"{selected_country}: Population Over Time"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------
# 🤖 Tab 4: Model Forecast
# ---------------------------
with tab4:
    st.subheader(f"{selected_country} — Forecasting Future Population (2021–2035)")

    X = country_df[["Year"]].values
    y = country_df["Population"].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(df["Year"].max() + 1, 2036)
    future_preds = model.predict(future_years.reshape(-1, 1))

    forecast_df = pd.DataFrame({"Year": future_years, "Predicted Population": future_preds})
    combined_df = pd.concat([
        country_df.rename(columns={"Population": "Actual Population"}).set_index("Year"),
        forecast_df.set_index("Year")
    ], axis=1).reset_index()

    fig_forecast = px.line(
        combined_df,
        x="Year",
        y=["Actual Population", "Predicted Population"],
        labels={"value": "Population", "variable": "Type"},
        color_discrete_map={"Actual Population": "violet", "Predicted Population": "purple"},
        title=f"{selected_country}: Population Forecast (2021–2035)"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ---------------------------
# 🔍 Tab 5: Multi-Country Comparison
# ---------------------------
with tab5:
    st.subheader("Multi-Country Population Comparison (1960–2035)")

    if len(multi_countries) == 0:
        st.warning("Please select at least one country to compare.")
    else:
        # Filter data for selected countries
        multi_df = df[df["Country Name"].isin(multi_countries)]

        # --- Trend Comparison ---
        st.markdown("### 📈 Historical Trends (1960–2020)")
        fig_multi_trend = px.line(
            multi_df,
            x="Year",
            y="Population",
            color="Country Name",
            markers=True,
            title="Population Trend Comparison (1960–2020)"
        )
        st.plotly_chart(fig_multi_trend, use_container_width=True)

        # --- Forecast Comparison ---
        st.markdown("### 🤖 Forecast Comparison (2021–2035)")
        forecast_combined = pd.DataFrame()

        for country in multi_countries:
            cdf = df[df["Country Name"] == country]
            X = cdf[["Year"]].values
            y = cdf["Population"].values

            model = LinearRegression()
            model.fit(X, y)

            future_years = np.arange(df["Year"].max() + 1, 2036)
            future_preds = model.predict(future_years.reshape(-1, 1))

            forecast_df = pd.DataFrame({
                "Country Name": country,
                "Year": future_years,
                "Predicted Population": future_preds
            })
            forecast_combined = pd.concat([forecast_combined, forecast_df])

        fig_multi_forecast = px.line(
            forecast_combined,
            x="Year",
            y="Predicted Population",
            color="Country Name",
            title="Population Forecast Comparison (2021–2035)"
        )
        st.plotly_chart(fig_multi_forecast, use_container_width=True)
