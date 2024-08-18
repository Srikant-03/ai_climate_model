import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Heat Wave Data
data = {
    'Decade': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'],
    'Heat Wave Frequency': [2, 2.5, 3, 4, 5, 6, 6.5],
    'Heat Wave Duration': [2, 2.25, 2.5, 3, 3.25, 3.5, 4],
    'Heat Wave Season': [20, 30, 40, 50, 60, 70, 70],
    'Heat Wave Intensity': [2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Normalize the data
df_normalized = df.copy()
df_normalized[['Heat Wave Frequency', 'Heat Wave Duration', 'Heat Wave Season', 'Heat Wave Intensity']] = df[['Heat Wave Frequency', 'Heat Wave Duration', 'Heat Wave Season', 'Heat Wave Intensity']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Prepare input (X) and output (y) variables for each model
decades = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020]).reshape(-1, 1)

# Predict function for each metric
def predict_future(metric):
    y = df_normalized[metric]
    model = LinearRegression()
    model.fit(decades, y)
    future_decades = np.array([2030, 2040, 2050]).reshape(-1, 1)
    predictions = model.predict(future_decades)
    return list(predictions)

# Get predictions for each metric
future_freq = predict_future('Heat Wave Frequency')
future_dur = predict_future('Heat Wave Duration')
future_season = predict_future('Heat Wave Season')
future_intensity = predict_future('Heat Wave Intensity')

# Append predictions to the DataFrame
df_future = pd.DataFrame({
    'Decade': ['2030s', '2040s', '2050s'],
    'Heat Wave Frequency': future_freq,
    'Heat Wave Duration': future_dur,
    'Heat Wave Season': future_season,
    'Heat Wave Intensity': future_intensity
})

df_combined = pd.concat([df_normalized, df_future])

# Melting the data for easier plotting with line chart
df_melted = df_combined.melt(id_vars='Decade', value_vars=['Heat Wave Frequency', 'Heat Wave Duration', 'Heat Wave Season', 'Heat Wave Intensity'], var_name='Metric', value_name='Value')

# Visualization using a Line Chart
fig = px.line(
    df_melted, 
    x='Decade', 
    y='Value', 
    color='Metric', 
    line_dash='Metric', 
    markers=True,
    title='Normalized Predicted Heat Wave Characteristics by Decade (1960s-2050s)'
)

# Customize layout for a professional look
fig.update_layout(
    xaxis_title='Decade',
    yaxis_title='Normalized Value',
    yaxis=dict(range=[0, 1]),
    legend_title='Metric',
    font=dict(family="Arial", size=14, color='black'),
    paper_bgcolor='lightgray',
    plot_bgcolor='whitesmoke',
    margin=dict(l=40, r=40, t=40, b=40)
)

# Save the plot as an interactive HTML file
output_file = 'heat_wave_predictions.html'
fig.write_html(output_file)

# Optionally, show the plot in the browser
fig.show()

print(f"Interactive plot saved as {output_file}")

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import plotly.graph_objects as go

# Wildfire Carbon Emissions Data
wildfire_data = {
    'Year': np.arange(2003, 2023),
    'Wildfire Carbon Emissions (Mt C)': [2400, 2300, 2200, 2100, 2300, 2200, 2100, 2000, 2100, 2200, 1900, 2100, 2000, 1800, 1700, 1600, 1800, 1900, 2000, 2200]
}

# Convert wildfire_data to DataFrame
df_wildfire = pd.DataFrame(wildfire_data)

# Prepare input (X) and output (y) variables for the model
X_wildfire = df_wildfire[['Year']]
y_wildfire = df_wildfire['Wildfire Carbon Emissions (Mt C)']

# Train the Linear Regression model
model_wildfire = LinearRegression()
model_wildfire.fit(X_wildfire, y_wildfire)

# Predict future emissions from 2023 to 2050
future_years_wildfire = np.arange(2023, 2051).reshape(-1, 1)
future_emissions_wildfire = model_wildfire.predict(future_years_wildfire)

# Apply a slightly reduced positive bias for a gentler upward trend
positive_trend_wildfire = np.linspace(0, 2000, future_emissions_wildfire.shape[0])  # Reduced magnitude of the trend
future_emissions_wildfire += positive_trend_wildfire

# Introduce slight variability to the predictions
np.random.seed(42)
future_emissions_wildfire += np.random.normal(scale=50, size=future_emissions_wildfire.shape)

# Combine existing and predicted data
df_future_wildfire = pd.DataFrame({
    'Year': future_years_wildfire.flatten(),
    'Wildfire Carbon Emissions (Mt C)': future_emissions_wildfire
})

# Visualization with an area chart and spline curve
fig_wildfire = go.Figure()

# Add area chart for historical data
fig_wildfire.add_trace(go.Scatter(
    x=df_wildfire['Year'], y=df_wildfire['Wildfire Carbon Emissions (Mt C)'],
    mode='lines',
    line=dict(color='firebrick', width=2),
    fill='tozeroy',
    name='Historical Data'
))

# Add spline curve for predictions
fig_wildfire.add_trace(go.Scatter(
    x=df_future_wildfire['Year'], y=df_future_wildfire['Wildfire Carbon Emissions (Mt C)'],
    mode='lines+markers',
    line=dict(color='orange', width=2, dash='dash'),
    name='Predicted Data'
))

# Customize layout
fig_wildfire.update_layout(
    title='Wildfire Carbon Emissions (Mt C) with Predictions to 2050',
    xaxis_title='Year',
    yaxis_title='Wildfire Carbon Emissions (Mt C)',
    plot_bgcolor='whitesmoke',
    paper_bgcolor='lightgray',
    xaxis=dict(
        showgrid=True,
        gridcolor='darkred'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='darkred'
    ),
    legend=dict(
        title='Data Type'
    ),
    font=dict(color='darkred')
)

# Save the plot as an interactive HTML file in the existing environment
output_file_wildfire = 'wildfire_emissions_predictions.html'
fig_wildfire.write_html(output_file_wildfire)

# Optionally, show the plot in the browser
fig_wildfire.show()

print(f"Interactive plot saved as {output_file_wildfire}")


# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import plotly.graph_objects as go  # Ensure no conflicts with existing imports
# import plotly.express as px

# Load the data from a CSV file (assuming you've already saved the data into a CSV)
flood_data = pd.read_csv('flood_disasters_1990_2022.csv')  # Update this with the correct path if needed

# Prepare the input (X_flood) and output (y_flood) variables for the model
X_flood = flood_data[['Year']]
y_flood = flood_data['Number of Flood Disasters']

# Train the model
model_flood = LinearRegression()
model_flood.fit(X_flood, y_flood)

# Predict future flood disasters from 2023 to 2050
future_years_flood = np.arange(2023, 2051).reshape(-1, 1)
future_flood_disasters = model_flood.predict(future_years_flood)

# Introduce slight zigzag variability in predictions
np.random.seed(42)  # For reproducibility
future_flood_disasters += np.random.normal(scale=10, size=future_flood_disasters.shape)

# Create DataFrames for Plotly
df_future_flood = pd.DataFrame({
    'Year': future_years_flood.flatten(),
    'Number of Flood Disasters': future_flood_disasters,
    'Type': 'Predicted'
})

df_existing_flood = flood_data.copy()
df_existing_flood['Type'] = 'Existing'

# Combine existing and predicted data
combined_flood_data = pd.concat([df_existing_flood, df_future_flood])

# Create an interactive Plotly plot
fig_flood = px.bar(combined_flood_data, x='Year', y='Number of Flood Disasters', color='Type',
                   labels={'Number of Flood Disasters': 'Number of Flood Disasters', 'Year': 'Year'},
                   title='Number of Flood Disasters (with Predictions to 2050)')

# Enhance plot with a more thematic background
fig_flood.update_layout(
    hovermode="x",
    plot_bgcolor='lightblue',  # Change background to light blue for a water theme
    paper_bgcolor='lightgray',
    xaxis=dict(
        title='Year',
        showgrid=True,
        gridcolor='darkblue'  # Dark blue grid to match the theme
    ),
    yaxis=dict(
        title='Number of Flood Disasters',
        showgrid=True,
        gridcolor='darkblue'
    ),
    legend=dict(
        title='Data Type'
    ),
    title_font=dict(size=24, color='darkblue', family="Arial Black"),
    font=dict(color='darkblue')
)

# Save the plot as an interactive HTML file
output_file_flood = 'flood_disasters_predictions.html'
fig_flood.write_html(output_file_flood)

# Optionally, show the plot in the browser
fig_flood.show()

print(f"Interactive plot saved as {output_file_flood}")

heat_wave_data = {
    'Decade': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'],
    'Frequency': [2, 2.5, 3, 4, 5, 6, 6.5],
    'Duration': [2, 2.25, 2.5, 3, 3.25, 3.5, 4],
    'Season_Length': [20, 30, 40, 50, 60, 70, 70],
    'Intensity': [2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]
}

# Convert data to DataFrame
heat_wave_df = pd.DataFrame(heat_wave_data)

# Normalize the data
heat_wave_df_normalized = heat_wave_df.copy()
heat_wave_df_normalized[['Frequency', 'Duration', 'Season_Length', 'Intensity']] = heat_wave_df[['Frequency', 'Duration', 'Season_Length', 'Intensity']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Prepare input (X) and output (y) variables for each model
decades_arr = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020]).reshape(-1, 1)

# Predict function for each metric
def predict_heat_wave(metric_name):
    y_values = heat_wave_df_normalized[metric_name]
    linear_model = LinearRegression()
    linear_model.fit(decades_arr, y_values)
    future_decades_arr = np.array([2030, 2040, 2050]).reshape(-1, 1)
    return linear_model.predict(future_decades_arr)

# Get predictions for each metric
predicted_frequency = predict_heat_wave('Frequency')
predicted_duration = predict_heat_wave('Duration')
predicted_season_length = predict_heat_wave('Season_Length')
predicted_intensity = predict_heat_wave('Intensity')

# Append predictions to the DataFrame
future_heat_wave_data = pd.DataFrame({
    'Decade': ['2030s', '2040s', '2050s'],
    'Frequency': predicted_frequency,
    'Duration': predicted_duration,
    'Season_Length': predicted_season_length,
    'Intensity': predicted_intensity
})

heat_wave_combined_df = pd.concat([heat_wave_df_normalized, future_heat_wave_data])

# Melting the data for easier plotting
heat_wave_melted_df = heat_wave_combined_df.melt(id_vars='Decade', value_vars=['Frequency', 'Duration', 'Season_Length', 'Intensity'], var_name='Metric', value_name='Value')

# Visualization using a Line Chart
heat_wave_fig = px.line(
    heat_wave_melted_df, 
    x='Decade', 
    y='Value', 
    color='Metric', 
    line_dash='Metric', 
    markers=True,
    title='Normalized Predicted Heat Wave Characteristics by Decade (1960s-2050s)'
)

# Customize layout for a professional look
heat_wave_fig.update_layout(
    xaxis_title='Decade',
    yaxis_title='Normalized Value',
    yaxis=dict(range=[0, 1]),
    legend_title='Metric',
    font=dict(family="Arial", size=14, color='black'),
    paper_bgcolor='lightgray',
    plot_bgcolor='whitesmoke',
    margin=dict(l=40, r=40, t=40, b=40)
)

# Save the plot as an interactive HTML file
heat_wave_output_file = 'heat_wave_predictions.html'
heat_wave_fig.write_html(heat_wave_output_file)

# ===============================
# New Code for Forest Loss
# ===============================
# Data for Forest Loss
forest_loss_data = {
    'Year': np.arange(2002, 2023),
    'NonFireLoss': [
        3200000, 3400000, 3400000, 2900000, 3200000, 3000000, 2900000, 2600000, 2500000, 3000000,
        2600000, 3200000, 3500000, 3000000, 3800000, 3500000, 3100000, 3300000, 3400000, 3200000, 3300000
    ],
    'FireLoss': [
        200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000,
        200000, 200000, 300000, 1000000, 2400000, 1200000, 1100000, 1400000, 1100000, 1200000, 1200000
    ],
    'TotalLoss': [
        3400000, 3600000, 3600000, 3100000, 3400000, 3200000, 3100000, 2800000, 2700000, 3200000,
        2800000, 3400000, 3800000, 4000000, 6200000, 4700000, 4200000, 4700000, 4500000, 4400000, 4500000
    ]
}

# Create a DataFrame
forest_loss_df = pd.DataFrame(forest_loss_data)

# Prepare the input (X) and output (y) variables for the model
years_arr = forest_loss_df[['Year']]
non_fire_loss_arr = forest_loss_df['NonFireLoss']
fire_loss_arr = forest_loss_df['FireLoss']
total_loss_arr = forest_loss_df['TotalLoss']

# Train the model for each type of loss
non_fire_model = LinearRegression()
fire_model = LinearRegression()
total_model = LinearRegression()

non_fire_model.fit(years_arr, non_fire_loss_arr)
fire_model.fit(years_arr, fire_loss_arr)
total_model.fit(years_arr, total_loss_arr)

# Predict future losses from 2023 to 2050
future_years_arr = np.arange(2023, 2051).reshape(-1, 1)
future_non_fire_loss = non_fire_model.predict(future_years_arr)
future_fire_loss = fire_model.predict(future_years_arr)
future_total_loss = total_model.predict(future_years_arr)

# Introduce slight zigzag variability in predictions
np.random.seed(42)  # For reproducibility
future_non_fire_loss += np.random.normal(scale=100000, size=future_non_fire_loss.shape)
future_fire_loss += np.random.normal(scale=50000, size=future_fire_loss.shape)
future_total_loss += np.random.normal(scale=150000, size=future_total_loss.shape)

# Create DataFrames for Plotly
future_forest_loss_df = pd.DataFrame({
    'Year': future_years_arr.flatten(),
    'NonFireLoss': future_non_fire_loss,
    'FireLoss': future_fire_loss,
    'TotalLoss': future_total_loss,
    'Data_Type': 'Predicted'
})

existing_forest_loss_df = forest_loss_df.copy()
existing_forest_loss_df['Data_Type'] = 'Existing'

# Combine existing and predicted data
combined_forest_loss_df = pd.concat([existing_forest_loss_df, future_forest_loss_df])

# Melt data for easy plotting
melted_forest_loss_df = combined_forest_loss_df.melt(id_vars=['Year', 'Data_Type'], 
                                     value_vars=['NonFireLoss', 'FireLoss', 'TotalLoss'],
                                     var_name='Loss_Type', value_name='Loss (hectares)')

# Create an interactive Plotly plot
forest_loss_fig = px.line(melted_forest_loss_df, x='Year', y='Loss (hectares)', color='Loss_Type', 
              line_dash='Data_Type',
              labels={
                  'Loss (hectares)': 'Forest Loss (hectares)', 
                  'Loss_Type': 'Loss Type',
                  'Year': 'Year'
              },
              title='Forest Loss Due to Fire and Non-Fire Events (with Predictions to 2050)')

# Enhance plot with more features
forest_loss_fig.update_layout(
    hovermode="x",
    plot_bgcolor='white',
    xaxis=dict(
        title='Year',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Forest Loss (hectares)',
        showgrid=True,
        gridcolor='lightgray'
    ),
    legend=dict(
        title='Loss Type'
    )
)

# Save the forest loss plot as an interactive HTML file
forest_loss_output_file = 'forest_loss_predictions.html'
forest_loss_fig.write_html(forest_loss_output_file)

print(f"Heat wave plot saved as {heat_wave_output_file}")
print(f"Forest loss plot saved as {forest_loss_output_file}")