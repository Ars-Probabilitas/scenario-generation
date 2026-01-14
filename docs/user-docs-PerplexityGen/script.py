
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic time series with known components
np.random.seed(42)
n = 200
t = np.arange(n)

# True components
true_trend = 0.05 * t + 10  # Linear trend
true_seasonal = 5 * np.sin(2 * np.pi * t / 20)  # Seasonal
true_noise = np.random.randn(n) * 1.5  # Random noise

# Combined series
y = true_trend + true_seasonal + true_noise

# Simulate different decomposition methods
# Method 1: Conditional expectation (rolling window)
window = 10
drift_conditional = np.zeros(n)
for i in range(1, n):
    start = max(0, i - window)
    drift_conditional[i] = np.mean(y[start:i])

martingale_conditional = y - drift_conditional

# Method 2: Exponential smoothing
alpha = 0.2
drift_exp = np.zeros(n)
drift_exp[0] = y[0]
for i in range(1, n):
    drift_exp[i] = alpha * y[i-1] + (1 - alpha) * drift_exp[i-1]

martingale_exp = y - drift_exp

# Create visualization data
data = {
    'Time': t.tolist(),
    'Original': y.tolist(),
    'True_Trend': true_trend.tolist(),
    'True_Noise': true_noise.tolist(),
    'Drift_Conditional': drift_conditional.tolist(),
    'Martingale_Conditional': martingale_conditional.tolist(),
    'Drift_Exp': drift_exp.tolist(),
    'Martingale_Exp': martingale_exp.tolist()
}

df = pd.DataFrame(data)
df.to_csv('decomposition_comparison_data.csv', index=False)

print("Data generated successfully")
print(f"Series length: {n}")
print(f"Original series range: [{y.min():.2f}, {y.max():.2f}]")
print(f"True trend range: [{true_trend.min():.2f}, {true_trend.max():.2f}]")
