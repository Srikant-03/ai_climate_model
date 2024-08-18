import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Data provided (years, mean CO2 levels, CO2 emissions in gigatons)
data = {
    'Year': np.arange(1979, 2023),
    'Mean CO2 (ppm)': [
        336.85, 338.91, 340.11, 340.86, 342.53, 344.07, 345.54, 346.97, 348.68, 351.16, 352.78, 354.05,
        355.39, 356.09, 356.83, 358.33, 360.17, 361.93, 363.05, 365.7, 367.8, 368.96, 370.57, 372.59,
        375.15, 376.95, 378.98, 381.15, 382.9, 385.02, 386.5, 388.76, 390.63, 392.65, 395.4, 397.34,
        399.65, 403.06, 405.22, 407.61, 410.07, 412.44, 414.7, 417.07,
    ],
    'CO2 Emission (Gigatons)': [
        0.11, 0.07, 0.09, 0.03, 0.06, 0.08, 0.07, 0.07, 0.1, 0.07, 0.07, 0.07, 0.07, 0.06, 0.07, 0.08,
        0.05, 0.04, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04, 0.04, 0.06, 0.05, 0.05, 0.04, 0.05, 0.04, 0.06,
        0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.07, 0.07, 0.07, 0.06, 0.07, 0.07,
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Filter data from 1980
df = df[df['Year'] >= 1980]

# Preparing the input (X) and output (y) variables for the model
X = df[['Year', 'Mean CO2 (ppm)']]
y = df['CO2 Emission (Gigatons)']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict future CO2 emissions from 2024 to 2050
future_years = np.arange(2024, 2051)
future_co2 = np.linspace(df['Mean CO2 (ppm)'].iloc[-1], df['Mean CO2 (ppm)'].iloc[-1] + (future_years[-1] - df['Year'].iloc[-1]) * 2.1, len(future_years))

# Combine future data into a DataFrame for prediction
future_X = pd.DataFrame({'Year': future_years, 'Mean CO2 (ppm)': future_co2})
future_pred = model.predict(future_X)

# Combine historical and predicted data for visualization
all_years = np.concatenate((df['Year'].values, future_X['Year'].values))
all_co2 = np.concatenate((df['Mean CO2 (ppm)'].values, future_co2))
all_emissions = np.concatenate((y, future_pred))

# Create a bubble chart
fig = go.Figure()

# Add bubbles for actual data
fig.add_trace(go.Scatter(
    x=df['Year'],
    y=df['Mean CO2 (ppm)'],
    mode='markers',
    marker=dict(
        size=df['CO2 Emission (Gigatons)'] * 100,  # Scale the size for visibility
        color=df['CO2 Emission (Gigatons)'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='CO2 Emissions (Gigatons)')
    ),
    name='Actual Data'
))

# Add bubbles for predicted data
fig.add_trace(go.Scatter(
    x=future_X['Year'],
    y=future_X['Mean CO2 (ppm)'],
    mode='markers',
    marker=dict(
        size=future_pred * 100,  # Scale the size for visibility
        color=future_pred,
        colorscale='Reds',
        showscale=False,
    ),
    name='Predicted Data'
))

# Update layout
fig.update_layout(
    title='Bubble Chart of CO2 Emissions vs. Year and Mean CO2 Levels',
    xaxis_title='Year',
    yaxis_title='Mean CO2 (ppm)',
    autosize=False,
    width=900,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90)
)

# Save the plot as an HTML file
fig.write_html('co2_emissions_bubble_chart.html')

print("Plot saved as 'co2_emissions_bubble_chart.html'")

# Save the complete data to an Excel file
output_file = "co2_emissions_1980_2050.xlsx"
df_future = pd.DataFrame({
    'Year': future_X['Year'],
    'Predicted CO2 Emission (Gigatons)': future_pred
})
complete_data = pd.concat([df, df_future], axis=0)
complete_data.to_excel(output_file, index=False)
