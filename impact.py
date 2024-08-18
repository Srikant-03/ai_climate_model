import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Data provided
data = {
    'Category': ['Data deficient', 'Least concern', 'Near threatened', 'Vulnerable', 
                 'Endangered', 'Critically endangered', 'Extinct or Extinct in the Wild'],
    'Count': [6584, 19032, 3931, 9075, 4891, 3325, 375],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define custom decline factors for each category to reflect the desired outcome
decline_factors = {
    'Data deficient': 0.4,          # More reduction
    'Least concern': 0.3,           # More reduction
    'Near threatened': 0.9,         # Less reduction
    'Vulnerable': 0.6,              # Moderate reduction
    'Endangered': 1.1,              # Slight increase
    'Critically endangered': 1.3,   # Increase
    'Extinct or Extinct in the Wild': 1.5  # Significant increase
}

# Apply the custom decline factors to simulate the predicted count for 2050
df['Predicted_Count_2050'] = df.apply(lambda row: row['Count'] * decline_factors[row['Category']], axis=1)

# Creating the two pie charts: one for the current year and one for 2050
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=['Biodiversity in 2023', 'Predicted Biodiversity in 2050'])

# Pie chart for 2023 with actual values and percentage share
fig.add_trace(go.Pie(
    labels=df['Category'],
    values=df['Count'],
    name="2023",
    textinfo='label+value+percent',  # Display label, value, and percentage
    hoverinfo='label+value+percent',
    textposition='inside'
), 1, 1)

# Pie chart for 2050 with adjusted actual values and percentage share
fig.add_trace(go.Pie(
    labels=df['Category'],
    values=df['Predicted_Count_2050'],
    name="2050",
    textinfo='label+value+percent',  # Display label, value, and percentage
    hoverinfo='label+value+percent',
    textposition='inside'
), 1, 2)

# Update layout
fig.update_layout(title_text='Biodiversity Comparison: 2023 vs 2050')

# Save the plot as an HTML file
fig.write_html("biodiversity_comparison_2023_vs_2050.html")

print("The graphs have been saved as 'biodiversity_comparison_2023_vs_2050.html'.")
