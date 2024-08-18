import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Data extracted from the graph (years 1980 to 2024)
years = np.array([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024])
mass_balance = np.array([-5, -7, -9, -11, -13, -15, -18, -21, -25, -27])

# Create a DataFrame to store the extracted data
data = pd.DataFrame({
    'Year': years,
    'Cumulative Glacier Mass Balance (m of water)': mass_balance
})

# Linear Regression Model to predict future values
model = LinearRegression()
model.fit(years.reshape(-1, 1), mass_balance)

# Predicting for years 2025 to 2050
future_years = np.arange(2025, 2051)
predicted_mass_balance = model.predict(future_years.reshape(-1, 1))

# Combine historical and predicted data into one DataFrame
future_data = pd.DataFrame({
    'Year': future_years,
    'Cumulative Glacier Mass Balance (m of water)': predicted_mass_balance
})
complete_data = pd.concat([data, future_data])

# Plotly Filled Area Plot
fig = go.Figure()

# Existing Data with blue fill above the line
fig.add_trace(go.Scatter(
    x=data['Year'],
    y=data['Cumulative Glacier Mass Balance (m of water)'],
    fill='tozeroy',
    fillcolor='rgba(0, 102, 204, 0.2)',  # Blue fill above the line
    line=dict(color='blue'),
    name='Existing Data',
    mode='lines'
))

# Predicted Data with red fill below the line
fig.add_trace(go.Scatter(
    x=future_data['Year'],
    y=future_data['Cumulative Glacier Mass Balance (m of water)'],
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.2)',  # Transparent red fill below the line
    line=dict(color='red', dash='dash'),
    name='Predicted Data',
    mode='lines'
))

# Update layout
fig.update_layout(
    title='Glacier Mass Balance (Yearly)',
    xaxis_title='Year',
    yaxis_title='Cumulative Glacier Mass Balance (meters of water)',
    template='plotly_dark',
    hovermode='x unified'
)

# Save the plot as an HTML file
fig.write_html('glacier_mass_balance_plot.html')

print("Plot saved as 'glacier_mass_balance_plot.html'")

# Save the complete data to an Excel file
output_file = "glacier_mass_balance_1980_2050.xlsx"
complete_data.to_excel(output_file, index=False)

print(f"Complete data saved to '{output_file}'")
