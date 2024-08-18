import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

# Data provided (years and AGGI values starting from 1980)
data = {
    'Year': np.arange(1980, 2023),
    'AGGI': [
        0.81, 0.83, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0,
        1.02, 1.03, 1.03, 1.05, 1.06, 1.08, 1.09, 1.11, 1.13, 1.14, 1.15, 1.16,
        1.18, 1.19, 1.21, 1.22, 1.23, 1.25, 1.26, 1.28, 1.29, 1.31, 1.33, 1.34,
        1.36, 1.38, 1.4, 1.42, 1.43, 1.45, 1.47, 1.49
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Prepare the input (X) and output (y) variables for the model
X = df[['Year']]
y = df['AGGI']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict future AGGI from 2023 to 2050
future_years = np.arange(2023, 2051).reshape(-1, 1)
future_aggis = model.predict(future_years)

# Create DataFrames for Plotly
df_future = pd.DataFrame({
    'Year': future_years.flatten(),
    'AGGI': future_aggis,
    'Type': 'Predicted'
})

df_existing = df.copy()
df_existing['Type'] = 'Existing'

# Combine existing and predicted data
combined_data = pd.concat([df_existing, df_future])

# Create an interactive Plotly plot
fig = px.line(combined_data, x='Year', y='AGGI', color='Type',
              labels={'AGGI': 'Annual Greenhouse Gas Index (AGGI)'},
              title='Annual Greenhouse Gas Index (AGGI) with Predictions to 2050')

# Enhance plot with more features
fig.update_layout(
    hovermode="x",
    plot_bgcolor='white',
    xaxis=dict(
        title='Year',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='AGGI',
        showgrid=True,
        gridcolor='lightgray'
    ),
    legend=dict(
        title='Data Type'
    )
)

# Save the plot as an HTML file
fig.write_html('aggi_plot.html')

print("Plot saved as 'aggi_plot.html'")

# Save the complete data to an Excel file
output_file = "aggi_1980_2050.xlsx"
complete_data = pd.concat([df_existing, df_future], axis=0)
complete_data.to_excel(output_file, index=False)

print(f"Data saved as '{output_file}'")
