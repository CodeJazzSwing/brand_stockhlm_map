#Dashboard with blue line around all datapoint.
# Bad example since ConvexHull does not hug around all the points.

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import plotly.graph_objects as go
import matplotlib.ticker as ticker

DATA_FILE = r'PATH_TO_DATA_FILE'
DATE_TIME = 'Tidpunkt_'
TARGET_COLUMN = 'Typ av brand'

df = pd.read_csv(DATA_FILE, sep=';')

# Stockholm stad logo
image = 'StockholmsStad_logotypeStandardA3_300ppi_svart.png'

# Convert DATE_TIME column from int64 to datetime object
df[DATE_TIME] = pd.to_datetime(df[DATE_TIME], format='%Y%m%d')

#---------SIDEBAR-------------

st.sidebar.header('Filter Here:')

# Date range filter
start_date = st.sidebar.date_input('Startdatum', df[DATE_TIME].min().date())
end_date = st.sidebar.date_input('Slutdatum', df[DATE_TIME].max().date())

# Error message for date
if start_date > end_date:
    st.sidebar.error('Misstag: Slutdatum måste vara efter startdatum.')


# Filter the DF based on the sidebar selection
df_selection = df[(df[DATE_TIME].dt.date >= start_date) & (df[DATE_TIME].dt.date <= end_date)]


# Create a new layout with 2 columns
col1, col2 = st.columns(2)

# Image on the top left side of map. 
col1.image(image, use_column_width=True)


# Create color dictionary för column "Typ of brand" categories
color_dict = {"Brand i byggnad" : "red", "Brand i container" : "blue", "Fordonsbrand" : "green", "Mark-/skogsbrand" : "orange", "Övrigt" : "purple"}
try:
    categories = df_selection[TARGET_COLUMN].unique()
except KeyError as e:
    print(f"Column, {TARGET_COLUMN}, is not in dataframe. Choose a column that is in the dataframe or check your datafile for the correct column.\n {e}")
    raise
colors = [color_dict[cat] for cat in categories]


# Use 'Typ av brand' for color parameter in the plot and set color_discrete_sequence to colors
fig = px.scatter_mapbox(df_selection,
                        lon = df_selection['lng'],
                        lat = df_selection['lat'],
                        zoom = 10,
                        width = 1000,
                        height= 1000,
                        title = 'Bränder i Stockholm stad',
                        text = df_selection[TARGET_COLUMN],
                        hover_data = {TARGET_COLUMN: True, DATE_TIME: True, 'lng': False, 'lat': False},
                        color = df_selection[TARGET_COLUMN],
                        color_discrete_sequence = colors
                        )


fig.update_traces(marker=dict(size = 12))
fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={'r':0,'t':50,'l':0,'b':10})


# Display the plot using Streamlit
st.plotly_chart(fig)


#_______________________LINE GRAPH___________________________
df_selection_grouped = df_selection.groupby([DATE_TIME, TARGET_COLUMN]).size().reset_index(name='counts')

# Get fire types and time points
fire_types = df_selection_grouped[TARGET_COLUMN].unique()
time_points = df_selection_grouped[DATE_TIME].unique()

# Create a DF with all combinations of time and fire
df_all_combinations = pd.DataFrame(index=pd.MultiIndex.from_product([time_points, fire_types], names=[DATE_TIME, TARGET_COLUMN])).reset_index()

# Merge the original DF with the DF that has all combinations
df_merged = pd.merge(df_all_combinations, df_selection_grouped, on=[DATE_TIME, TARGET_COLUMN], how='left')

# Fill NA values with 0
df_merged['counts'] = df_merged['counts'].fillna(0)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Width of a bar 
width = 0.1

# Generate bars for each fire type
for i, fire_type in enumerate(fire_types):
    # Filter the data for the current fire type
    df_fire_type = df_merged[df_merged[TARGET_COLUMN] == fire_type]
    
    # Create an array for the position of each bar on the x-axis
    r = np.arange(len(time_points))
    
    # Plot the data
    ax.bar(r + i*width, df_fire_type['counts'], width=width, label=fire_type)

# Set the title and labels
ax.set_title('Bränder i Stockholm stad')
ax.set_xlabel(DATE_TIME)
ax.set_ylabel('Antal bränder')

# Add xticks on the middle of the group bars
ax.set_xticks(r + width/2, time_points)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Add a legend
ax.legend()

# Use MaxNLocator for y-axis
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Display the plot
st.pyplot(fig)

#______________Create a line around the map________________OBS! This is a test.

# Create the scatter mapbox for the points
fig = go.Figure(go.Scattermapbox(
    lon = df_selection['lng'],
    lat = df_selection['lat'],
    mode = 'markers',
    marker = go.scattermapbox.Marker(
        size = 12,
        color = df_selection[TARGET_COLUMN].map(color_dict),
        colorscale = colors
    ),
    text = df_selection[TARGET_COLUMN],
))

# Calculate the Convex Hull of the points
points = df_selection[['lat', 'lng']].values
hull = ConvexHull(points)

# Get the coordinates of the Convex Hull vertices
hull_points = points[hull.vertices]

# Create a new trace for the Convex Hull boundary
hull_trace = go.Scattermapbox(
    lon = np.append(hull_points[:, 1], hull_points[0, 1]),  # Append the first point to the end to close the polygon
    lat = np.append(hull_points[:, 0], hull_points[0, 0]),
    mode = 'lines',
    line = dict(width = 2, color = 'blue'),
    name = 'Convex Hull'
)

# Add the new trace to the figure
fig.add_trace(hull_trace)

# Set the layout
fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken='pk.eyJ1Ijoibm9lbHN3ZWNvIiwiYSI6ImNscDZ4ZzI0dDF5Z2syaHF1dnR6eGxmMWwifQ.fCT2l0c2VFiJmOFoAU-xMA',
        bearing=0,
        center=dict(
            lat=df_selection['lat'].mean(),
            lon=df_selection['lng'].mean()
        ),
        pitch=0,
        zoom=10,
        style='basic'  # or any other Mapbox style
    ),
)

# Display the plot in Streamlit
st.plotly_chart(fig)
