import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_excel('Book1.xlsx')  # Replace with your file path

# Filter the data from 1980 onwards
filtered_data = data[data['Year'] >= 1980]

# Prepare the data for modeling
years_array = filtered_data['Year'].values.reshape(-1, 1)
temperature_values = filtered_data['Temperature'].values

# Create and train the model
linear_model = LinearRegression()
linear_model.fit(years_array, temperature_values)

# Make predictions for the given range
predicted_temperatures = linear_model.predict(years_array)

# Make predictions for future years
future_years_array = np.array(range(2024, 2051)).reshape(-1, 1)
future_temperature_predictions = linear_model.predict(future_years_array)

# Create a DataFrame for future predictions
future_prediction_data = pd.DataFrame({
    'Year': future_years_array.flatten(),
    'Temperature': future_temperature_predictions
})

# Create a scatter plot with animation frames
fig = go.Figure()

# Add initial scatter traces for observed and fitted data
fig.add_trace(go.Scatter(x=filtered_data['Year'], y=temperature_values, mode='lines', name='Observed Data', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=filtered_data['Year'], y=predicted_temperatures, mode='lines', name='Fitted Line', line=dict(color='red')))

# Add future predictions as a scatter trace
fig.add_trace(go.Scatter(x=future_prediction_data['Year'], y=future_temperature_predictions, mode='lines', name='Future Predictions', line=dict(color='green', dash='dash')))

# Create frames for animation
frames = []
for i in range(1, len(filtered_data) + len(future_prediction_data) + 1):
    frame_data = {
        'data': [
            {
                'x': filtered_data['Year'][:i],
                'y': temperature_values[:i],
                'mode': 'lines',
                'name': 'Observed Data',
                'line': {'color': 'blue'}
            },
            {
                'x': filtered_data['Year'][:i],
                'y': predicted_temperatures[:i],
                'mode': 'lines',
                'name': 'Fitted Line',
                'line': {'color': 'red'}
            },
            {
                'x': future_prediction_data['Year'][:max(0, i - len(filtered_data))],
                'y': future_temperature_predictions[:max(0, i - len(filtered_data))],
                'mode': 'lines',
                'name': 'Future Predictions',
                'line': {'color': 'green', 'dash': 'dash'}
            }
        ]
    }
    frames.append(frame_data)

# Add frames to figure
fig.update(frames=frames)

# Set up the layout
fig.update_layout(
    title='Temperature Over Time and Future Predictions (Since 1980)',
    xaxis_title='Year',
    yaxis_title='Temperature (°C)',
    xaxis=dict(range=[1980, 2050], showgrid=False, zeroline=False),
    yaxis=dict(range=[min(temperature_values) - 0.5, max(future_temperature_predictions) + 0.5], showgrid=False, zeroline=False),
    shapes=[dict(type='line', x0=1980, x1=2050, y0=13.8, y1=13.8, line=dict(color='orange', dash='dash'))],
    annotations=[dict(x=1980, y=13.8, xref='x', yref='y', text='Suitable Temperature for Earth', showarrow=True, arrowhead=2)],
    margin=dict(l=0, r=0, t=50, b=0),
    height=600,
    autosize=True,
    showlegend=True
)

# Add JavaScript for auto-starting and looping animation
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {
                    'frame': {'duration': 100, 'redraw': True},
                    'fromcurrent': True,
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': 'Play',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }],
    sliders=[]
)

# Save the figure as an HTML file
fig.write_html('temperature_predictions.html')

print("Animated plot saved as 'temperature_predictions.html'")
