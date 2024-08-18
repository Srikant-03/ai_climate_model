import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Load data from the CSV files
comparison_data = pd.read_csv("disaster_comparison_1980_2000.csv")

# Prepare data for linear regression for each disaster type
disaster_types = ['Floods', 'Cyclones', 'Earthquakes']
models = {}
predictions = {}

for disaster in disaster_types:
    X = comparison_data['Year'].values.reshape(-1, 1)
    y = comparison_data[disaster].values
    model = LinearRegression()
    model.fit(X, y)
    models[disaster] = model
    
    # Predict future values and adjust to start where the historical data ends
    future_years = np.arange(2001, 2051).reshape(-1, 1)
    prediction = model.predict(future_years)
    
    # Scale predictions to match the ending point of historical data
    prediction = prediction + (y[-1] - prediction[0])
    
    # Add zigzag pattern to predictions
    noise = np.sin(future_years.flatten() / 2) * np.std(y) * 0.3  # Zigzag pattern
    predictions[disaster] = prediction + noise

# Visualization of the data
fig = go.Figure()

# Offset flood data slightly to avoid overlap with cyclones
flood_offset = 0.05 * comparison_data['Floods'].max()

# Plot historical data and future predictions using different visualization types

# Floods
fig.add_trace(go.Scatter(
    x=comparison_data['Year'],
    y=comparison_data['Floods'] - flood_offset,
    mode='lines+markers',
    marker=dict(color='cyan', size=6),
    line=dict(color='cyan', width=3),
    name='Floods (1980-2000)',
))

fig.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predictions['Floods'] - flood_offset,
    mode='lines',
    fill='tozeroy',
    line=dict(color='cyan', width=2, dash='dot'),
    name='Floods Prediction (2001-2050)',
    opacity=0.5
))

# Cyclones
fig.add_trace(go.Scatter(
    x=comparison_data['Year'],
    y=comparison_data['Cyclones'],
    mode='lines+markers',
    marker=dict(color='green', size=6),
    line=dict(color='green', width=3),
    name='Cyclones (1980-2000)',
))

fig.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predictions['Cyclones'],
    mode='lines',
    fill='tozeroy',
    line=dict(color='green', width=2, dash='dot'),
    name='Cyclones Prediction (2001-2050)',
    opacity=0.5
))

# Earthquakes
fig.add_trace(go.Scatter(
    x=comparison_data['Year'],
    y=comparison_data['Earthquakes'],
    mode='lines+markers',
    marker=dict(color='red', size=6),
    line=dict(color='red', width=3),
    name='Earthquakes (1980-2000)',
))

fig.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predictions['Earthquakes'],
    mode='lines',
    fill='tozeroy',
    line=dict(color='red', width=2, dash='dot'),
    name='Earthquakes Prediction (2001-2050)',
    opacity=0.5
))

# Update layout for a more appealing look
fig.update_layout(
    title='Climatic Disasters vs Earthquakes (1980-2050)',
    xaxis_title='Year',
    yaxis_title='Number of Events',
    template='plotly_white',
    hovermode='x unified',
    legend_title_text='Disaster Type',
    font=dict(size=16),
    plot_bgcolor='rgba(240, 240, 240, 1)',
    paper_bgcolor='rgba(240, 240, 240, 1)',
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.5)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.5)'
    )
)

# Save the plot as an HTML file
fig.write_html('climatic_disasters_vs_earthquakes.html')

# Save the predictions and data to an Excel file
output_data = pd.DataFrame({
    'Year': np.concatenate([comparison_data['Year'].values, future_years.flatten()]),
    'Floods': np.concatenate([comparison_data['Floods'].values, predictions['Floods'].flatten()]),
    'Cyclones': np.concatenate([comparison_data['Cyclones'].values, predictions['Cyclones'].flatten()]),
    'Earthquakes': np.concatenate([comparison_data['Earthquakes'].values, predictions['Earthquakes'].flatten()])
})
output_file = "climatic_disasters_vs_earthquakes_1980_2050.xlsx"
output_data.to_excel(output_file, index=False)

print("Plot saved as 'climatic_disasters_vs_earthquakes.html'")
print(f"Data saved to '{output_file}'")
