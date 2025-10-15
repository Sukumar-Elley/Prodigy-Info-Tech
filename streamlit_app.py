# ============================================================
# 🌍 Streamlit Global Population Dashboard (1960–2020 + Forecast)
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Global Population Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ============================================================
# Data Loading with Streamlit Cache
# ============================================================
@st.cache_resource
def load_data():
    path = "data/API_SP.POP.TOTL_DS2_en_csv_v2_23043.csv"
    df = pd.read_csv(path, skiprows=4)
    df = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="Year",
        value_name="Population"
    )
    df = df.dropna(subset=["Population"])
    df["Year"] = df["Year"].astype(int)
    return df

df = load_data()
st.sidebar.success("✅ Data loaded successfully!")

# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.header("🔧 Controls")
year = st.sidebar.slider("Select Year", 1960, 2020, 2020)
country = st.sidebar.selectbox("Select Country", sorted(df["Country Name"].unique()))

# ============================================================
# Tabs Layout
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 World Map",
    "📊 Top 10 Countries",
    "📈 Country Trend",
    "🔮 Model Forecast"
])

# ============================================================
# 🌍 TAB 1 — Animated World Choropleth
# ============================================================
with tab1:
    st.subheader("Global Population Animation (1960–2020)")

    df_filtered = df[df["Year"] == year]
    fig = px.choropleth(
        df,
        locations="Country Code",
        color="Population",
        hover_name="Country Name",
        animation_frame="Year",
        color_continuous_scale="Viridis",
        title="🌍 World Population Growth (1960–2020)",
        projection="natural earth"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Download data ---
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Data (CSV)",
        data=csv,
        file_name=f"world_population_{year}.csv",
        mime="text/csv"
    )

    # --- Download plot ---
    st.download_button(
        "📸 Download Plot (PNG)",
        data=fig.to_image(format="png"),
        file_name=f"world_map_{year}.png",
        mime="image/png"
    )

# ============================================================
# 📊 TAB 2 — Top 10 Countries
# ============================================================
with tab2:
    st.subheader(f"Top 10 Most Populous Countries in {year}")

    top10 = df[df["Year"] == year].nlargest(10, "Population")
    fig2 = px.bar(
        top10,
        x="Country Name",
        y="Population",
        text="Population",
        color="Country Name",
        title=f"Top 10 Countries by Population ({year})"
    )
    fig2.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "📥 Download Data (CSV)",
        data=top10.to_csv(index=False).encode("utf-8"),
        file_name=f"top10_{year}.csv",
        mime="text/csv"
    )

# ============================================================
# 📈 TAB 3 — Country Trend Line
# ============================================================
with tab3:
    st.subheader(f"{country} Population Trend (1960–2020)")

    df_country = df[df["Country Name"] == country]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_country["Year"],
        y=df_country["Population"],
        mode="lines+markers",
        name="Population"
    ))
    fig3.update_layout(title=f"{country} Population Trend (1960–2020)",
                       xaxis_title="Year",
                       yaxis_title="Population")
    st.plotly_chart(fig3, use_container_width=True)

    st.download_button(
        "📥 Download Trend Data (CSV)",
        data=df_country.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_trend.csv",
        mime="text/csv"
    )

# ============================================================
# 🔮 TAB 4 — Model Forecast
# ============================================================
with tab4:
    st.subheader(f"Forecast Next 10 Years for {country}")

    df_country = df[df["Country Name"] == country]
    X = df_country[["Year"]]
    y = np.log1p(df_country["Population"])

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(2021, 2031).reshape(-1, 1)
    preds = np.expm1(model.predict(future_years))

    forecast_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted Population": preds
    })

    fig4 = px.line(
        forecast_df,
        x="Year",
        y="Predicted Population",
        markers=True,
        title=f"{country} — Forecasted Population (2021–2030)"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.download_button(
        "📥 Download Forecast Data (CSV)",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_forecast.csv",
        mime="text/csv"
    )

# ============================================================
# End of App
# ============================================================
st.success("✅ Dashboard ready! Explore all tabs above.")
