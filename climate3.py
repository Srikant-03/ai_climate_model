import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Load the data
data = pd.read_csv('seawater.csv')  # Replace with your file path

# Start analysis from 1980
start_year = 1980

# Filter the data based on the starting year
filtered_data = data[data['Year'] >= start_year]

# Prepare the data for modeling
X = filtered_data['Year'].values.reshape(-1, 1)
y = filtered_data['SeaLevelRise'].values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions for future years
future_years = np.arange(2021, 2051).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Combine historical and future data for visualization
all_years = np.concatenate((filtered_data['Year'].values, future_years.flatten()))
all_sea_levels = np.concatenate((y, future_predictions))

# Create a bar chart
bar_chart = go.Bar(
    x=all_years,
    y=all_sea_levels,
    name='Sea Level Rise',
    marker=dict(color=all_sea_levels, colorscale='Viridis'),
    hoverinfo='x+y'
)

# Create a heatmap for visual emphasis
heatmap = go.Heatmap(
    x=all_years,
    y=['Sea Level Rise'] * len(all_years),  # Single row heatmap
    z=[all_sea_levels],  # Ensure the z parameter is 2D
    colorscale='Viridis',
    showscale=False,
    opacity=0.6,
    hoverinfo='none'
)

# Combine the bar chart and heatmap
fig = go.Figure(data=[heatmap, bar_chart])

# Update the layout
fig.update_layout(
    title='Sea Level Rise Over Time and Future Predictions (Since 1980)',
    xaxis_title='Year',
    yaxis_title='Sea Level Rise (mm)',
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    showlegend=False
)

# Save the plot as an HTML file
fig.write_html('sea_level_rise_plot.html')

print("Plot saved as 'sea_level_rise_plot.html'")

# Output future predictions (optional)
for year, rise in zip(future_years.flatten(), future_predictions):
    print(f"Year: {year}, Predicted Sea Level Rise: {rise:.2f} mm")
