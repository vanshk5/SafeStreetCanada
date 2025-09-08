# -------------------------------
# SafeStreet Canada - Accident Risk Dashboard
# Deployment-ready for Streamlit Cloud
# -------------------------------

# -------------------------------
# Robust Imports
# -------------------------------
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import altair as alt
    import pydeck as pdk
    from pathlib import Path
except ImportError as e:
    missing_package = str(e).split()[-1]
    raise ImportError(
        f"Missing package: {missing_package}. Please add it to requirements.txt and redeploy the app."
    )

# -------------------------------
# Base directory and file paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / 'accident_model.pkl'
FEATURES_PATH = BASE_DIR / 'features_columns.pkl'
DATA_PATH = BASE_DIR / 'processed_data_numeric.csv'

# -------------------------------
# Load model & features
# -------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} not found. Push the model file to GitHub.")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"{FEATURES_PATH} not found. Push the features file to GitHub.")

model = joblib.load(MODEL_PATH)
features_columns = joblib.load(FEATURES_PATH)

# -------------------------------
# Load data
# -------------------------------
if not DATA_PATH.exists():
    st.warning(f"{DATA_PATH} not found. Some app features may not work.")
    data = pd.DataFrame(columns=features_columns)  # empty dataframe fallback
else:
    data = pd.read_csv(DATA_PATH)

# -------------------------------
# Identify Neighborhood Columns
# -------------------------------
neigh_cols = [col for col in data.columns if col.startswith('NEIGHBOURHOOD_')]

if neigh_cols:
    data['NEIGHBOURHOOD'] = data[neigh_cols].idxmax(axis=1)
    data['NEIGHBOURHOOD'] = data['NEIGHBOURHOOD'].str.replace(r'NEIGHBOURHOOD_\d+_', '', regex=True)
    data['NEIGHBOURHOOD'] = data['NEIGHBOURHOOD'].str.replace(r'\s*\(\d+\)', '', regex=True)
else:
    st.warning("No neighborhood columns found in data.")

# -------------------------------
# Make Predictions
# -------------------------------
if all(col in data.columns for col in features_columns):
    X = data[features_columns]
    data['Severe_Accident_Probability'] = model.predict_proba(X)[:, 1] * 100
else:
    st.warning("Required feature columns not found. Predictions cannot be made.")
    data['Severe_Accident_Probability'] = 0

# -------------------------------
# Top 20 Most Accident-Prone Neighborhoods
# -------------------------------
top_areas = (
    data.groupby('NEIGHBOURHOOD')['Severe_Accident_Probability']
    .mean()
    .reset_index()
    .sort_values(by='Severe_Accident_Probability', ascending=False)
    .head(20)
)

# Color code: top 5 red, rest blue
top_areas['color'] = ['red' if i < 5 else 'blue' for i in range(len(top_areas))]

# -------------------------------
# Neighborhood Coordinates (replace with accurate lat/lon)
# -------------------------------
neigh_coords = {
    "Agincourt South-Malvern West": (43.801, -79.246),
    "Alderwood": (43.634, -79.556),
    # Add all neighborhoods you want
}

top_areas['lat'] = top_areas['NEIGHBOURHOOD'].map(lambda x: neigh_coords.get(x, 43.65107))  # Toronto center fallback
top_areas['lon'] = top_areas['NEIGHBOURHOOD'].map(lambda x: neigh_coords.get(x, -79.347015))

# -------------------------------
# Streamlit Layout
# -------------------------------
st.title("🚨 SafeStreet Canada - Toronto Accident Risk Dashboard")
st.markdown("""
This dashboard shows the **most accident-prone neighborhoods in Toronto** based on historical accident data.
Top 5 neighborhoods are highlighted in red. Map shows locations with risk-based colors and circle sizes.
""")

# -------------------------------
# Bar Chart
# -------------------------------
chart = alt.Chart(top_areas).mark_bar().encode(
    x=alt.X('Severe_Accident_Probability:Q', title='Probability of Severe Accident (%)'),
    y=alt.Y('NEIGHBOURHOOD:N', sort='-x', title='Neighborhood'),
    color=alt.Color('color:N', scale=None, legend=None),
    tooltip=['NEIGHBOURHOOD', alt.Tooltip('Severe_Accident_Probability', format=".2f")]
).properties(
    width=700,
    height=500,
    title="Top 20 Accident-Prone Neighborhoods"
)
st.altair_chart(chart)

# -------------------------------
# Table
# -------------------------------
st.subheader("Detailed Probabilities")
st.dataframe(top_areas[['NEIGHBOURHOOD', 'Severe_Accident_Probability']].reset_index(drop=True).style.format({
    'Severe_Accident_Probability': "{:.2f}%"
}))

# -------------------------------
# Map Visualization
# -------------------------------
st.subheader("Neighborhood Map")

# Normalize probability for circle radius
max_radius = 800
min_radius = 200
prob_min = top_areas['Severe_Accident_Probability'].min()
prob_max = top_areas['Severe_Accident_Probability'].max()
top_areas['radius'] = top_areas['Severe_Accident_Probability'].apply(
    lambda x: min_radius + (x - prob_min) / (prob_max - prob_min) * (max_radius - min_radius)
)

# Color gradient: blue (low) -> red (high)
top_areas['color_map'] = top_areas['Severe_Accident_Probability'].apply(
    lambda x: [int(255 * (x - prob_min) / (prob_max - prob_min)), 0, int(255 * (prob_max - x) / (prob_max - prob_min)), 160]
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v10',
    initial_view_state=pdk.ViewState(
        latitude=43.65107,
        longitude=-79.347015,
        zoom=10,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=top_areas,
            get_position='[lon, lat]',
            get_color='color_map',
            get_radius='radius',
            pickable=True,
        ),
    ],
    tooltip={"text": "{NEIGHBOURHOOD}\n{Severe_Accident_Probability} %"}
))

