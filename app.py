import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")

# ──────────────────────────────────────────────
# 1. Load & Train Model (cached so it runs once)
# ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_excel("AQI_Dataset.xlsx")
    df = pd.get_dummies(df, columns=["City"], drop_first=True)
    df_processed = df.fillna(0)

    y = df_processed["AQI"]
    X = df_processed.drop(columns=["AQI", "AQI_Bucket", "Date",
                                    "City_Chennai", "City_Delhi",
                                    "City_Hyderabad", "City_Mumbai"],
                           errors="ignore")

    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, X.columns.tolist(), r2

model, feature_cols, r2 = train_model()

# ──────────────────────────────────────────────
# 2. UI
# ──────────────────────────────────────────────
st.title("🌫️ AQI Prediction App")
st.markdown("Enter pollutant values below to predict the **Air Quality Index (AQI)**.")
st.caption(f"Model R² (training): **{r2:.4f}**")

st.divider()

city = st.selectbox("🏙️ City", ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai"])

col1, col2, col3 = st.columns(3)
with col1:
    pm25    = st.number_input("PM2.5",    min_value=0.0, value=30.0, step=0.1)
    no      = st.number_input("NO",       min_value=0.0, value=9.0,  step=0.1)
    nox     = st.number_input("NOx",      min_value=0.0, value=25.0, step=0.1)
    co      = st.number_input("CO",       min_value=0.0, value=1.0,  step=0.01)
    benzene = st.number_input("Benzene",  min_value=0.0, value=1.0,  step=0.01)
with col2:
    pm10    = st.number_input("PM10",     min_value=0.0, value=70.0, step=0.1)
    no2     = st.number_input("NO2",      min_value=0.0, value=27.0, step=0.1)
    nh3     = st.number_input("NH3",      min_value=0.0, value=13.0, step=0.1)
    so2     = st.number_input("SO2",      min_value=0.0, value=8.0,  step=0.1)
    toluene = st.number_input("Toluene",  min_value=0.0, value=36.0, step=0.1)
with col3:
    o3      = st.number_input("O3",       min_value=0.0, value=37.0, step=0.1)
    xylene  = st.number_input("Xylene",   min_value=0.0, value=0.0,  step=0.01)

st.divider()

if st.button("🔍 Predict AQI", use_container_width=True):

    # Build input with same features the model was trained on
    # The notebook dropped city dummies for Chennai/Delhi/Hyderabad/Mumbai,
    # leaving Bengaluru as the baseline (drop_first on alphabetically sorted dummies).
    pollution_intensity_index = 0  # not in raw dataset; filled with 0

    base = {
        "PM2.5": pm25, "PM10": pm10, "NO": no,
        "NO2 Traffic index": no2, "NOx": nox, "NH3": nh3,
        "CO": co, "SO2": so2, "O3": o3,
        "Benzene": benzene, "Toluene": toluene, "Xylene": xylene,
        "Pollution Intensity Index": pollution_intensity_index,
    }

    # Build a dict aligned to whatever columns the model actually has
    input_dict = {}
    for col in feature_cols:
        input_dict[col] = base.get(col, 0)

    input_df = pd.DataFrame([input_dict])
    aqi = model.predict(input_df)[0]

    # ── AQI Category ──────────────────────────────────
    if aqi <= 100:
        category, color, emoji = "Moderate",  "#2ECC71", "🟢"
    elif aqi <= 200:
        category, color, emoji = "Average",   "#F39C12", "🟡"
    elif aqi <= 300:
        category, color, emoji = "High",      "#E67E22", "🟠"
    else:
        category, color, emoji = "Extreme",   "#E74C3C", "🔴"

    st.markdown(
        f"""
        <div style='background:{color}22; border-left:6px solid {color};
                    border-radius:8px; padding:20px; margin-bottom:16px;'>
            <h2 style='margin:0; color:{color}'>{emoji} Predicted AQI: {aqi:.2f}</h2>
            <p style='margin:4px 0 0; font-size:1.1em; color:{color}'>
                AQI Level: <strong>{category}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── City-Specific Recommendations ─────────────────
    st.subheader(f"📋 Recommendations for {city}")

    recs = {
        "Delhi": {
            "Individual":     ["Avoid outdoor exposure during peak hours",
                                "Use N95 masks and air purifiers"],
            "Organizational": ["Implement work-from-home (WFH) policies",
                                "Provide indoor air quality monitoring"],
            "Educational":    ["Shift to online classes during high AQI days",
                                "Limit outdoor sports activities"],
            "Government":     ["Enforce odd-even vehicle policy",
                                "Ban construction activities temporarily",
                                "Promote EV adoption and stricter emission norms"],
        },
        "Mumbai": {
            "Individual":     ["Avoid high-traffic areas during peak hours",
                                "Use masks in polluted zones"],
            "Organizational": ["Reduce outdoor work exposure",
                                "Implement flexible working hours"],
            "Educational":    ["Conduct indoor activities",
                                "Raise awareness on air pollution"],
            "Government":     ["Strict control on construction dust",
                                "Enforce regulations on urban emissions"],
        },
        "Chennai": {
            "Individual":     ["Continue normal activities with awareness",
                                "Use public transport when possible"],
            "Organizational": ["Encourage eco-friendly commuting",
                                "Maintain green office spaces"],
            "Educational":    ["Promote environmental education programs",
                                "Encourage tree plantation drives"],
            "Government":     ["Maintain green cover and coastal advantage",
                                "Invest in sustainable urban planning"],
        },
        "Bengaluru": {
            "Individual":     ["Reduce personal vehicle usage",
                                "Prefer carpooling or public transport"],
            "Organizational": ["Implement hybrid work models",
                                "Provide transport facilities"],
            "Educational":    ["Limit outdoor activities during peak traffic",
                                "Promote sustainability awareness"],
            "Government":     ["Improve traffic management systems",
                                "Expand metro and public transport"],
        },
        "Hyderabad": {
            "Individual":     ["Avoid industrial zones during peak hours",
                                "Use protective masks"],
            "Organizational": ["Monitor employee exposure in industrial areas",
                                "Implement flexible working policies"],
            "Educational":    ["Conduct awareness sessions on pollution",
                                "Encourage indoor activities"],
            "Government":     ["Monitor industrial emissions strictly",
                                "Control urban expansion and pollution zones"],
        },
    }

    city_recs = recs.get(city, {})
    if city_recs:
        for level, tips in city_recs.items():
            with st.expander(f"🔹 {level} Level"):
                for tip in tips:
                    st.markdown(f"- {tip}")
    else:
        st.info("No specific recommendations available for this city.")
