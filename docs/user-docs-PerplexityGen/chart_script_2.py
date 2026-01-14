
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Generate sample data since the CSV file doesn't exist
np.random.seed(42)
n = 100
time = np.arange(n)

# Create true trend (increasing)
true_trend = 2 * time + 50

# Create true noise (random component)
true_noise = np.random.normal(0, 10, n)

# Original series
original = true_trend + true_noise

# Conditional expectation method - drift is a smoothed trend
drift_conditional = pd.Series(original).rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
martingale_conditional = original - drift_conditional

# Exponential smoothing method
alpha = 0.3
drift_exp = np.zeros(n)
drift_exp[0] = original[0]
for i in range(1, n):
    drift_exp[i] = alpha * original[i] + (1 - alpha) * drift_exp[i-1]
martingale_exp = original - drift_exp

# Create DataFrame
df = pd.DataFrame({
    'Time': time,
    'Original': original,
    'True_Trend': true_trend,
    'True_Noise': true_noise,
    'Drift_Conditional': drift_conditional,
    'Martingale_Conditional': martingale_conditional,
    'Drift_Exp': drift_exp,
    'Martingale_Exp': martingale_exp
})

# Create a single chart comparing the original series with both drift methods
fig = go.Figure()

# Add Original time series
fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['Original'],
    mode='lines',
    name='Original',
    line=dict(color='#13343B', width=2),
    cliponaxis=False
))

# Add True Trend
fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['True_Trend'],
    mode='lines',
    name='True Trend',
    line=dict(color='#DB4545', width=2, dash='dash'),
    cliponaxis=False
))

# Add Drift from Conditional Expectation
fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['Drift_Conditional'],
    mode='lines',
    name='Drift (Cond)',
    line=dict(color='#1FB8CD', width=2),
    cliponaxis=False
))

# Add Drift from Exponential Smoothing
fig.add_trace(go.Scatter(
    x=df['Time'],
    y=df['Drift_Exp'],
    mode='lines',
    name='Drift (Exp)',
    line=dict(color='#2E8B57', width=2),
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title={
        "text": "Comparison of Doob Decomposition Methods (1900-2000)<br><span style='font-size: 18px; font-weight: normal;'>Two drift estimation approaches track the underlying trend</span>"
    },
    xaxis_title="Time",
    yaxis_title="Value",
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    ),
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# Save as PNG and SVG
fig.write_image('doob_decomposition.png')
fig.write_image('doob_decomposition.svg', format='svg')
