import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd

# Data extracted from the image
years = np.array([1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990,
                  1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                  2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

ice_extent = np.array([7.0, 6.9, 6.7, 7.1, 7.1, 6.8, 6.9, 6.8, 6.7, 6.8, 6.6, 6.8,
                       6.6, 6.5, 6.3, 6.4, 6.3, 6.3, 6.4, 6.1, 6.1, 6.1, 5.8, 6.1,
                       5.9, 6.0, 5.8, 5.7, 5.6, 5.4, 5.3, 5.5, 5.1, 4.8, 5.0, 4.8,
                       4.9, 5.3, 4.7, 4.5, 4.3, 4.7, 4.3, 4.6, 4.5])

# Reshape the data for the linear regression model
X = years.reshape(-1, 1)
y = ice_extent

# Create the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict future values (up to 2050)
future_years = np.arange(2024, 2051).reshape(-1, 1)
predicted_ice_extent = model.predict(future_years)

# Create a combined dataset for visualization
all_years = np.concatenate([years, future_years.flatten()])
all_ice_extent = np.concatenate([ice_extent, predicted_ice_extent])

# Create a 2D density plot
fig = go.Figure(go.Histogram2dContour(
    x=all_years,
    y=all_ice_extent,
    colorscale='Blues',  # Color scale from light blue to dark blue
    contours=dict(coloring='heatmap'),
    histnorm='density',  # Normalize the histogram by the total number of samples
    line=dict(width=0),
    showscale=False  # Remove the color scale/legend from the side
))

# Add a scatter trace for observed and predicted data points
fig.add_trace(go.Scatter(
    x=years,
    y=ice_extent,
    mode='markers',
    marker=dict(color='blue', size=8),
    name='Observed Data'
))

fig.add_trace(go.Scatter(
    x=future_years.flatten(),
    y=predicted_ice_extent,
    mode='markers',
    marker=dict(color='red', size=8, symbol='diamond'),
    name='Predicted Data'
))

# Update layout
fig.update_layout(
    title='Observed and Predicted Arctic Sea Ice Extent (Density Plot)',
    xaxis_title='Year',
    yaxis_title='Sea Ice Extent (million km²)',
    template='plotly_dark',
    hovermode='x unified'
)

# Save the plot as an HTML file
fig.write_html('arctic_sea_ice_extent_plot.html')

print("Plot saved as 'arctic_sea_ice_extent_plot.html'")

# Save the complete data to an Excel file
complete_data = pd.DataFrame({
    'Year': np.concatenate([years, future_years.flatten()]),
    'Sea Ice Extent (million km²)': np.concatenate([ice_extent, predicted_ice_extent])
})
output_file = "arctic_sea_ice_extent_1979_2050.xlsx"
complete_data.to_excel(output_file, index=False)

print(f"Complete data saved to '{output_file}'")
