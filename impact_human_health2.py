import pandas as pd
import plotly.graph_objects as go

# Data provided
data = {
    "City": ["San Diego", "Spokane", "Billings", "Winnipeg", "Montreal", 
             "Chicago", "Boston", "Dallas", "Jacksonville"],
    "Current Weeks": [12, 0, 0, 0, 0, 0, 4, 20, 30],
    "+2°C Weeks": [20, 8, 12, 12, 12, 10, 10, 30, 36],
    "+4°C Weeks": [30, 16, 20, 20, 20, 20, 20, 40, 40]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a grouped bar chart
fig = go.Figure()

# Add bars for each temperature scenario
fig.add_trace(go.Bar(
    x=df['City'],
    y=df['Current Weeks'],
    name='Current Weeks',
    marker=dict(color='#1f77b4')
))

fig.add_trace(go.Bar(
    x=df['City'],
    y=df['+2°C Weeks'],
    name='+2°C Weeks',
    marker=dict(color='#ff7f0e')
))

fig.add_trace(go.Bar(
    x=df['City'],
    y=df['+4°C Weeks'],
    name='+4°C Weeks',
    marker=dict(color='#2ca02c')
))

# Update layout for better aesthetics
fig.update_layout(
    title={
        'text': 'Dengue Transmission Period by City and Temperature Increase',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 22, 'color': '#ffffff'}
    },
    xaxis_title='City',
    yaxis_title='Weeks of Transmission',
    barmode='group',  # Grouped bars
    template='plotly_dark',  # Dark theme for a modern look
    margin=dict(l=60, r=60, t=80, b=60),  # Adjusted margins for a cleaner layout
    legend=dict(title="Temperature Scenario"),
    hovermode='x unified'  # Unified hover mode for better interactivity
)

# Save the plot as an HTML file
file_path = 'dengue_transmission_period_plot.html'
fig.write_html(file_path)

print(f"The plot has been saved as '{file_path}'.")
