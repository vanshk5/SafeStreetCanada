import streamlit as st
import pandas as pd
import joblib
import altair as alt
import pydeck as pdk

# --------------------------
# Constants / File Paths
# --------------------------
MODEL_PATH = 'accident_model.pkl'
DATA_PATH = 'processed_data_numeric.csv'
FEATURES_PATH = 'features_columns.pkl'

# --------------------------
# Load Model & Features
# --------------------------
model = joblib.load(MODEL_PATH)
features_columns = joblib.load(FEATURES_PATH)

# --------------------------
# Load Data
# --------------------------
data = pd.read_csv(DATA_PATH)

# --------------------------
# Identify Neighborhood Columns
# --------------------------
neigh_cols = [col for col in data.columns if col.startswith('NEIGHBOURHOOD_')]

# Create a single 'NEIGHBOURHOOD' column for display
data['NEIGHBOURHOOD'] = data[neigh_cols].idxmax(axis=1)
# Remove prefix
data['NEIGHBOURHOOD'] = data['NEIGHBOURHOOD'].str.replace(r'NEIGHBOURHOOD_\d+_', '', regex=True)
# Remove numbers in parentheses
data['NEIGHBOURHOOD'] = data['NEIGHBOURHOOD'].str.replace(r'\s*\(\d+\)', '', regex=True)

# --------------------------
# Make Predictions
# --------------------------
X = data[features_columns]
data['Severe_Accident_Probability'] = model.predict_proba(X)[:, 1] * 100  # percentage

# --------------------------
# Top 20 Most Accident-Prone Neighborhoods
# --------------------------
top_areas = (
    data.groupby('NEIGHBOURHOOD')['Severe_Accident_Probability']
    .mean()
    .reset_index()
    .sort_values(by='Severe_Accident_Probability', ascending=False)
    .head(20)
)

# Color code: top 5 in red, rest in blue (for bar chart)
top_areas['color'] = ['red' if i < 5 else 'blue' for i in range(len(top_areas))]

# --------------------------
# Neighborhood Coordinates (sample, replace with accurate lat/lon)
# --------------------------
neigh_coords = {
    "Agincourt South-Malvern West": (43.801, -79.246),
    "Alderwood": (43.634, -79.556),
    # Add all neighborhoods you want
}

# Map top areas to coordinates
top_areas['lat'] = top_areas['NEIGHBOURHOOD'].map(lambda x: neigh_coords.get(x, 43.65107))  # default Toronto center
top_areas['lon'] = top_areas['NEIGHBOURHOOD'].map(lambda x: neigh_coords.get(x, -79.347015))

# --------------------------
# Streamlit App Layout
# --------------------------
st.title("🚨 SafeStreetCanada - Accident Risk Dashboard")
st.markdown("""
This dashboard shows the **most accident-prone neighborhoods in Toronto** based on historical accident data.
The top 5 neighborhoods are highlighted in red. Map shows their locations with risk-based colors and circle sizes.
""")

# --------------------------
# Bar Chart
# --------------------------
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

# --------------------------
# Table
# --------------------------
st.subheader("Detailed Probabilities")
st.dataframe(top_areas[['NEIGHBOURHOOD', 'Severe_Accident_Probability']].reset_index(drop=True).style.format({
    'Severe_Accident_Probability': "{:.2f}%"
}))

# --------------------------
# Map Visualization
# --------------------------
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
