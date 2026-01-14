# Doob Decomposition with TinyTimeMixer (TTM) - User Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Getting Started](#getting-started)
4. [Installation](#installation)
5. [Core Concepts](#core-concepts)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Introduction

This documentation covers the implementation of **Doob Decomposition** applied to time series forecasting using **TinyTimeMixer (TTM)** models. The notebook `doob_decomp_ttm.ipynb` demonstrates how to decompose time series data into martingale and predictable components, then leverage TTM's lightweight architecture for enhanced forecasting performance.

### What You'll Learn
- How to apply Doob decomposition to time series data
- Using TinyTimeMixer for zero-shot and few-shot forecasting
- Combining stochastic process theory with modern deep learning
- Interpreting martingale and drift components in forecasts

### Prerequisites
- Basic understanding of time series analysis
- Familiarity with Python and Jupyter notebooks
- Knowledge of stochastic processes (helpful but not required)

---

## Theoretical Background

### Doob Decomposition Theorem

The **Doob Decomposition Theorem** is a fundamental result in stochastic process theory that states any adapted and integrable stochastic process can be uniquely decomposed into two components:

**Mathematical Formulation:**

For a discrete-time adapted process \( X = (X_n)_{n \in I} \), there exists a unique decomposition:

\[ X_n = M_n + A_n \]

where:
- \( M = (M_n) \) is a **martingale** (unpredictable component)
- \( A = (A_n) \) is a **predictable process** (drift component) with \( A_0 = 0 \)

**Key Properties:**
- **Martingale Component**: \( \mathbb{E}[M_{n+1} | \mathcal{F}_n] = M_n \)
- **Predictable Component**: \( A_n \) is \( \mathcal{F}_{n-1} \)-measurable
- **Uniqueness**: The decomposition is almost surely unique

### Why Doob Decomposition for Forecasting?

1. **Noise Separation**: Isolates systematic trends (drift) from random fluctuations (martingale)
2. **Interpretability**: Provides clear understanding of predictable vs. unpredictable dynamics
3. **Enhanced Modeling**: Allows separate treatment of deterministic and stochastic components
4. **Risk Assessment**: Martingale component quantifies inherent uncertainty

### TinyTimeMixer (TTM) Overview

**TinyTimeMixer** is a compact pre-trained foundation model for time series forecasting developed by IBM Research. Key features include:

#### Architecture Highlights
- **Non-Transformer Design**: Uses MLP-Mixer architecture (no attention mechanism)
- **Compact Size**: Starting from 1M parameters
- **Multi-level Modeling**: Captures both univariate and multivariate dependencies
- **Adaptive Patching**: Dynamically adjusts patch sizes across layers
- **Resolution Prefix Tuning**: Handles multiple time series frequencies

#### Key Innovations
1. **Channel-Independent Pre-training**: Learns temporal dynamics from univariate series
2. **Channel Mixing**: Activated during fine-tuning for multivariate correlations
3. **Exogenous Mixing**: Incorporates known future variables
4. **Zero-Shot Forecasting**: Pre-trained on 244M samples for immediate deployment

---

## Getting Started

### Quick Start Example

```python
import numpy as np
import pandas as pd
from src.doob_decomposition import DoobDecomposer
from src.ttm_forecaster import TTMForecaster

# Load your time series data
data = pd.read_csv('your_timeseries.csv')

# Step 1: Apply Doob Decomposition
decomposer = DoobDecomposer()
martingale, drift = decomposer.decompose(data['values'])

# Step 2: Initialize TTM model
ttm_model = TTMForecaster(
    model_path="ibm/TTM",
    context_length=512,
    prediction_length=96
)

# Step 3: Forecast using decomposed components
forecast = ttm_model.predict(
    martingale_component=martingale,
    drift_component=drift
)

# Step 4: Visualize results
decomposer.plot_decomposition(data['values'], martingale, drift)
```

---

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install torch>=2.0.0
pip install transformers>=4.30.0

# TinyTimeMixer specific
pip install tsfm_public
pip install datasets

# Visualization
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# Optional: For GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Installation from Repository

```bash
# Clone the repository
git clone https://github.com/Ars-Probabilitas/scenario-generation.git
cd scenario-generation

# Install in development mode
pip install -e .

# Navigate to examples
cd examples/TinyTimeMixers
jupyter notebook doob_decomp_ttm.ipynb
```

---

## Core Concepts

### 1. Doob Decomposition Process

#### Step-by-Step Breakdown

**Input**: Time series \( X_t \), \( t = 1, 2, ..., T \)

**Step 1: Compute Drift Component**

\[ A_n = \sum_{k=1}^{n} \left( \mathbb{E}[X_k | \mathcal{F}_{k-1}] - X_{k-1} \right) \]

**Step 2: Compute Martingale Component**

\[ M_n = X_0 + \sum_{k=1}^{n} \left( X_k - \mathbb{E}[X_k | \mathcal{F}_{k-1}] \right) \]

**Step 3: Verify Decomposition**

\[ X_n = M_n + A_n \quad \forall n \]

#### Implementation Considerations

- **Conditional Expectation Estimation**: Use rolling windows, exponential smoothing, or ML models
- **Filtration**: Define information set carefully (e.g., past observations, exogenous variables)
- **Stationarity**: Check for and handle non-stationary series

### 2. TTM Integration Strategy

#### Multi-Level Modeling Approach

**Phase 1: Pre-training (Univariate)**
```
Input → Normalization → Patching → TTM Backbone → TTM Decoder → Forecast Head
```

**Phase 2: Fine-tuning (Multivariate)**
```
Input → Channel Mixing → Exogenous Mixing → Frozen Backbone → Updated Decoder → Output
```

#### Adaptive Patching

The TTM backbone uses hierarchical patch processing:

```
Level 1: [c, n, hf] → [c, 4n, hf/4] → Mixing → [c, n, hf]
Level 2: [c, n, hf] → [c, 16n, hf/16] → Mixing → [c, n, hf]
Level 3: [c, n, hf] → [c, 64n, hf/64] → Mixing → [c, n, hf]
```

This allows the model to learn patterns at multiple temporal resolutions.

---

## Usage Guide

### Basic Workflow

#### 1. Data Preparation

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('timeseries_data.csv', parse_dates=['timestamp'])

# Handle missing values
df = df.fillna(method='ffill')

# Create features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Split data
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]
```

#### 2. Doob Decomposition

```python
from src.doob_decomposition import DoobDecomposer

# Initialize decomposer
decomposer = DoobDecomposer(
    method='conditional_expectation',  # or 'adaptive', 'exponential_smoothing'
    window_size=24,
    alpha=0.3  # for exponential smoothing
)

# Decompose training data
train_results = decomposer.fit_transform(train_data['value'].values)

# Extract components
martingale_train = train_results['martingale']
drift_train = train_results['drift']

# Decompose test data
test_results = decomposer.transform(test_data['value'].values)
```

#### 3. TTM Model Configuration

```python
from transformers import TinyTimeMixerForPrediction

# Load pre-trained model
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm/TTM",
    revision="main"
)

# Configure for your task
config = {
    "context_length": 512,      # Input sequence length
    "prediction_length": 96,    # Forecast horizon
    "num_channels": 1,          # Univariate or multivariate
    "patch_length": 64,         # Initial patch size
    "num_layers": 8,            # Number of mixing layers
    "dropout": 0.1,
    "resolution_prefix": "daily"  # 'hourly', 'daily', 'weekly', etc.
}
```

#### 4. Training / Fine-tuning

```python
from torch.utils.data import DataLoader
from src.ttm_trainer import TTMTrainer

# Create dataset with decomposed components
train_dataset = create_ttm_dataset(
    original=train_data['value'].values,
    martingale=martingale_train,
    drift=drift_train,
    context_length=512,
    prediction_length=96
)

# Initialize trainer
trainer = TTMTrainer(
    model=model,
    train_dataset=train_dataset,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=50,
    use_drift_component=True,
    use_martingale_component=True
)

# Fine-tune model
trainer.train()
```

#### 5. Forecasting

```python
# Generate forecasts
forecasts = model.predict(
    context=test_data['value'][:512].values,
    martingale_context=test_results['martingale'][:512],
    drift_context=test_results['drift'][:512]
)

# Post-process predictions
forecast_df = pd.DataFrame({
    'timestamp': test_data['timestamp'][512:512+96],
    'forecast': forecasts['mean'],
    'lower_bound': forecasts['quantile_0.1'],
    'upper_bound': forecasts['quantile_0.9']
})
```

### Advanced Usage

#### Custom Conditional Expectation Estimator

```python
from sklearn.ensemble import GradientBoostingRegressor

class CustomDoobDecomposer(DoobDecomposer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.estimator = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5
        )
    
    def estimate_conditional_expectation(self, X, y, X_pred):
        """Custom conditional expectation using ML model"""
        self.estimator.fit(X, y)
        return self.estimator.predict(X_pred)

# Use custom decomposer
custom_decomposer = CustomDoobDecomposer()
results = custom_decomposer.fit_transform(data)
```

#### Multi-Channel Forecasting with Exogenous Variables

```python
# Prepare multivariate data
multivariate_data = pd.DataFrame({
    'target': df['sales'],
    'exog_1': df['price'],
    'exog_2': df['promotion'],
    'exog_3': df['weather']
})

# Decompose each channel
decomposed_channels = {}
for col in multivariate_data.columns:
    decomposed_channels[col] = decomposer.fit_transform(
        multivariate_data[col].values
    )

# Configure TTM for multivariate
config['num_channels'] = 4
config['enable_channel_mixing'] = True
config['enable_exogenous_mixing'] = True

# Train
model = train_multivariate_ttm(
    channels=decomposed_channels,
    config=config
)
```

---

## API Reference

### DoobDecomposer Class

```python
class DoobDecomposer:
    """
    Performs Doob decomposition on time series data.
    
    Parameters
    ----------
    method : str, default='conditional_expectation'
        Method for estimating conditional expectations:
        - 'conditional_expectation': Use rolling window averaging
        - 'exponential_smoothing': Exponential weighted average
        - 'adaptive': Adaptive window sizing
        
    window_size : int, default=24
        Size of rolling window for conditional expectation estimation
        
    alpha : float, default=0.3
        Smoothing parameter for exponential smoothing (0 < alpha < 1)
        
    min_periods : int, default=1
        Minimum number of observations required for estimation
    """
    
    def __init__(self, method='conditional_expectation', 
                 window_size=24, alpha=0.3, min_periods=1):
        pass
    
    def fit_transform(self, X):
        """
        Fit decomposer and transform data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to decompose
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'martingale': Martingale component
            - 'drift': Predictable drift component
            - 'original': Original series
            - 'verification': X - (M + A) should be ~0
        """
        pass
    
    def transform(self, X):
        """Apply fitted decomposer to new data."""
        pass
    
    def plot_decomposition(self, X, martingale, drift, 
                          figsize=(15, 10), save_path=None):
        """Visualize decomposition results."""
        pass
```

### TTMForecaster Class

```python
class TTMForecaster:
    """
    Wrapper for TinyTimeMixer forecasting with Doob decomposition support.
    
    Parameters
    ----------
    model_path : str
        Path to pre-trained TTM model or HuggingFace model ID
        
    context_length : int
        Length of input context window
        
    prediction_length : int
        Number of steps to forecast
        
    use_decomposition : bool, default=True
        Whether to use Doob decomposition components
        
    device : str, default='cuda'
        Device for model inference ('cuda' or 'cpu')
    """
    
    def __init__(self, model_path, context_length, 
                 prediction_length, use_decomposition=True,
                 device='cuda'):
        pass
    
    def predict(self, context, martingale_component=None, 
                drift_component=None, return_quantiles=True):
        """
        Generate forecasts using TTM model.
        
        Parameters
        ----------
        context : array-like
            Historical time series data
            
        martingale_component : array-like, optional
            Martingale component from Doob decomposition
            
        drift_component : array-like, optional
            Drift component from Doob decomposition
            
        return_quantiles : bool, default=True
            Whether to return prediction intervals
            
        Returns
        -------
        forecasts : dict
            Dictionary with keys:
            - 'mean': Point forecasts
            - 'quantile_0.1', 'quantile_0.9': Prediction intervals
            - 'martingale_forecast': Forecast of martingale component
            - 'drift_forecast': Forecast of drift component
        """
        pass
    
    def evaluate(self, y_true, y_pred, metrics=['mse', 'mae', 'mape']):
        """Compute forecast accuracy metrics."""
        pass
```

### Helper Functions

```python
def create_ttm_dataset(original, martingale, drift, 
                       context_length, prediction_length):
    """
    Create PyTorch dataset for TTM training.
    
    Parameters
    ----------
    original : array-like
        Original time series
    martingale : array-like
        Martingale component
    drift : array-like
        Drift component
    context_length : int
        Input sequence length
    prediction_length : int
        Forecast horizon
        
    Returns
    -------
    dataset : torch.utils.data.Dataset
        Dataset object for DataLoader
    """
    pass

def compute_metrics(y_true, y_pred, metrics=['mse', 'mae', 'mape', 'smape']):
    """
    Compute multiple forecast accuracy metrics.
    
    Returns
    -------
    results : dict
        Dictionary of metric names and values
    """
    pass
```

---

## Examples

### Example 1: Basic Doob Decomposition and Forecasting

```python
import numpy as np
import pandas as pd
from src.doob_decomposition import DoobDecomposer
from src.ttm_forecaster import TTMForecaster
import matplotlib.pyplot as plt

# Generate synthetic time series with trend and noise
np.random.seed(42)
t = np.arange(1000)
trend = 0.05 * t  # Linear trend
seasonal = 10 * np.sin(2 * np.pi * t / 50)  # Seasonal pattern
noise = np.random.randn(1000) * 2  # Random noise
y = trend + seasonal + noise

# Step 1: Apply Doob decomposition
decomposer = DoobDecomposer(method='exponential_smoothing', alpha=0.2)
results = decomposer.fit_transform(y)

# Visualize decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

axes[0].plot(y, label='Original', color='black')
axes[0].set_title('Original Time Series')
axes[0].legend()

axes[1].plot(results['martingale'], label='Martingale (Random)', color='blue')
axes[1].set_title('Martingale Component')
axes[1].legend()

axes[2].plot(results['drift'], label='Drift (Predictable)', color='red')
axes[2].set_title('Drift Component')
axes[2].legend()

axes[3].plot(results['verification'], label='Residual (X - M - A)', color='green')
axes[3].set_title('Verification (should be ~0)')
axes[3].legend()

plt.tight_layout()
plt.savefig('doob_decomposition.png')

# Step 2: Forecast using TTM
forecaster = TTMForecaster(
    model_path='ibm/TTM',
    context_length=512,
    prediction_length=96
)

# Use last 512 points as context
context = y[-512:]
mart_context = results['martingale'][-512:]
drift_context = results['drift'][-512:]

# Generate forecast
forecast = forecaster.predict(
    context=context,
    martingale_component=mart_context,
    drift_component=drift_context
)

print(f"Forecast shape: {forecast['mean'].shape}")
print(f"Forecast range: [{forecast['mean'].min():.2f}, {forecast['mean'].max():.2f}]")
```

### Example 2: Multivariate Time Series with Exogenous Variables

```python
# Load electricity dataset
df = pd.read_csv('electricity.csv', parse_dates=['timestamp'])

# Prepare target and exogenous variables
target = df['consumption'].values
exog_temp = df['temperature'].values
exog_hour = df['hour'].values

# Decompose target
decomposer = DoobDecomposer(window_size=24)
target_decomp = decomposer.fit_transform(target)

# Decompose exogenous variables
exog_decomp = {
    'temperature': decomposer.fit_transform(exog_temp),
    'hour': decomposer.fit_transform(exog_hour)
}

# Configure multivariate TTM
from transformers import TinyTimeMixerConfig

config = TinyTimeMixerConfig(
    context_length=168,  # 1 week of hourly data
    prediction_length=24,  # Forecast next day
    num_input_channels=3,  # Target + 2 exogenous
    enable_channel_mixing=True,
    enable_exogenous_mixing=True
)

# Prepare multivariate input
X_train = np.stack([
    target_decomp['martingale'],
    exog_decomp['temperature']['martingale'],
    exog_decomp['hour']['martingale']
], axis=1)

# Train model
# ... training code ...

# Forecast
forecast_multivariate = forecaster.predict(
    context=X_train[-168:],
    num_channels=3
)
```

### Example 3: Zero-Shot Forecasting on New Domain

```python
# Load pre-trained TTM model
from transformers import TinyTimeMixerForPrediction

model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2"
)

# Your new domain data (e.g., stock prices)
stock_data = pd.read_csv('stock_prices.csv')

# Apply Doob decomposition
decomposer = DoobDecomposer(method='adaptive', window_size=20)
stock_decomp = decomposer.fit_transform(stock_data['close'].values)

# Zero-shot forecast (no fine-tuning)
context = stock_decomp['martingale'][-512:]

with torch.no_grad():
    forecast = model.generate(
        past_values=torch.tensor(context).unsqueeze(0).unsqueeze(-1),
        prediction_length=30
    )

# Plot results
plt.figure(figsize=(15, 6))
plt.plot(range(len(context)), context, label='Historical', color='blue')
plt.plot(range(len(context), len(context)+30), 
         forecast[0, :, 0].numpy(), 
         label='Forecast', color='red', linestyle='--')
plt.title('Zero-Shot Stock Price Forecast with Doob Decomposition')
plt.legend()
plt.show()
```

### Example 4: Comparison of Methods

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Test different decomposition methods
methods = ['conditional_expectation', 'exponential_smoothing', 'adaptive']
results_comparison = {}

for method in methods:
    # Decompose
    decomposer = DoobDecomposer(method=method)
    decomp = decomposer.fit_transform(y_train)
    
    # Forecast
    forecaster = TTMForecaster(
        model_path='ibm/TTM',
        context_length=512,
        prediction_length=96
    )
    
    forecast = forecaster.predict(
        context=y_train[-512:],
        martingale_component=decomp['martingale'][-512:],
        drift_component=decomp['drift'][-512:]
    )
    
    # Evaluate
    mse = mean_squared_error(y_test[:96], forecast['mean'])
    mae = mean_absolute_error(y_test[:96], forecast['mean'])
    
    results_comparison[method] = {
        'MSE': mse,
        'MAE': mae
    }

# Display results
comparison_df = pd.DataFrame(results_comparison).T
print(comparison_df)
```

---

## Best Practices

### 1. Data Preparation

**Do:**
- ✅ Check for and handle missing values before decomposition
- ✅ Normalize/standardize data appropriately
- ✅ Split data chronologically (no random shuffling)
- ✅ Ensure sufficient history for conditional expectation estimation

**Don't:**
- ❌ Use future information in training (data leakage)
- ❌ Apply decomposition separately on train/test (fit on train, transform on test)
- ❌ Ignore outliers without investigation
- ❌ Mix different frequencies without resampling

### 2. Doob Decomposition

**Choosing Window Size:**
- Short-term patterns (hourly/daily): 24-168 observations
- Medium-term (weekly): 4-12 weeks
- Long-term (monthly/yearly): 12-36 periods

**Method Selection:**
- **Conditional Expectation**: Good for stationary series
- **Exponential Smoothing**: Better for trending data
- **Adaptive**: Best for non-stationary with regime changes

### 3. TTM Configuration

**Context Length:**
- Minimum: 2× seasonal period
- Recommended: 4-8× seasonal period
- Maximum: Model-dependent (usually 512-1024)

**Prediction Length:**
- Should be << context length
- Typical ratios: context/prediction ∈ [4, 10]

**Fine-tuning Strategy:**
- Start with zero-shot to establish baseline
- Fine-tune only if performance is insufficient
- Freeze backbone, update decoder only (faster, less overfitting)

### 4. Model Evaluation

**Metrics:**
- MSE/RMSE: Sensitive to large errors
- MAE: Robust to outliers
- MAPE: Scale-independent, interpretable
- SMAPE: Symmetric version of MAPE

**Validation Strategy:**
- Use expanding window or rolling window cross-validation
- Never use k-fold CV on time series
- Maintain temporal order

### 5. Production Deployment

**Performance:**
- TTM is CPU-friendly (unlike Transformers)
- Batch predictions when possible
- Cache decomposition results for reuse

**Monitoring:**
- Track forecast accuracy over time
- Monitor drift in martingale/drift components
- Set up alerts for distribution shifts

---

## Troubleshooting

### Common Issues

#### Issue 1: Poor Decomposition Quality

**Symptoms:**
- Martingale component shows clear trends
- Drift component is nearly zero
- Verification residuals are large

**Solutions:**
```python
# Increase window size for smoother drift estimation
decomposer = DoobDecomposer(window_size=48)  # Instead of 24

# Try different estimation method
decomposer = DoobDecomposer(method='exponential_smoothing', alpha=0.1)

# Check for non-stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print(f'ADF Statistic: {result[0]}, p-value: {result[1]}')

# If non-stationary, difference first
y_diff = np.diff(y)
decomp = decomposer.fit_transform(y_diff)
```

#### Issue 2: TTM Model Not Loading

**Error:** `OSError: Model not found`

**Solutions:**
```python
# Check model path
from transformers import AutoConfig
config = AutoConfig.from_pretrained("ibm/TTM")  # Test connection

# Use local model if offline
model_path = "./local_models/TTM"
model = TinyTimeMixerForPrediction.from_pretrained(model_path, local_files_only=True)

# Check token for private models
from huggingface_hub import login
login(token="your_token_here")
```

#### Issue 3: Poor Forecast Accuracy

**Diagnosis:**
```python
# Check each component separately
forecast_full = forecaster.predict(context, mart_context, drift_context)
forecast_martingale_only = forecaster.predict(context, mart_context, None)
forecast_drift_only = forecaster.predict(context, None, drift_context)

# Compare errors
print(f"Full model MSE: {compute_mse(y_test, forecast_full['mean'])}")
print(f"Martingale only MSE: {compute_mse(y_test, forecast_martingale_only['mean'])}")
print(f"Drift only MSE: {compute_mse(y_test, forecast_drift_only['mean'])}")
```

**Solutions:**
- Fine-tune model on domain-specific data
- Adjust context/prediction lengths
- Include exogenous variables if available
- Ensemble multiple models

#### Issue 4: Memory Issues with Large Datasets

**Solutions:**
```python
# Process in chunks
chunk_size = 10000
all_martingales = []
all_drifts = []

for i in range(0, len(y), chunk_size):
    chunk = y[i:i+chunk_size]
    decomp = decomposer.transform(chunk)
    all_martingales.append(decomp['martingale'])
    all_drifts.append(decomp['drift'])

# Use data generators for training
def data_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# Reduce model precision
model = model.half()  # FP16 instead of FP32
```

---

## References

### Theoretical Foundations

1. **Doob, J. L.** (1953). *Stochastic Processes*. Wiley.
   - Original formulation of the Doob decomposition theorem

2. **Meyer, P. A.** (1962). "Decomposition of supermartingales: the uniqueness theorem." *Illinois Journal of Mathematics*.
   - Extension to continuous-time processes

3. **Williams, D.** (1991). *Probability with Martingales*. Cambridge University Press.
   - Accessible introduction to martingale theory

### TinyTimeMixer

4. **Ekambaram, V., et al.** (2024). "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series." *arXiv:2401.03955*.
   - Original TTM paper

5. **Chen, S. I., et al.** (2023). "TSMixer: An All-MLP Architecture for Time Series Forecasting." *arXiv:2303.06053*.
   - Foundation architecture for TTM

### Applications

6. **Heath, D. C., & Jackson, P. L.** (1994). "Modeling the evolution of demand forecasts with application to safety stock analysis." *IIE Transactions*.
   - Martingale model for forecast evolution

7. **Foster, D. P., & Stine, R. A.** (2021). "Threshold Martingales and the Evolution of Forecasts." *arXiv:2105.06834*.
   - Modern applications in forecasting

### Software & Tools

8. **IBM Granite Time Series Foundation Models** - https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2
9. **sktime Documentation** - https://www.sktime.net/
10. **TinyTimeMixer GitHub** - https://github.com/ibm-granite/granite-tsfm

---

## Appendix

### A. Mathematical Details

#### Proof Sketch: Uniqueness of Doob Decomposition

Suppose we have two decompositions:
\[ X_n = M_n + A_n = M'_n + A'_n \]

Then:
\[ M_n - M'_n = A'_n - A_n \]

The left side is a martingale, the right side is predictable. A process that is both a martingale and predictable must be constant. Since \( A_0 = A'_0 = 0 \), we have \( A_n = A'_n \) and \( M_n = M'_n \) almost surely.

#### Computation of Conditional Expectation

For discrete processes with finite history:
\[ \mathbb{E}[X_t | \mathcal{F}_{t-1}] = \mathbb{E}[X_t | X_{t-1}, X_{t-2}, ..., X_0] \]

Practical estimation via rolling window:
\[ \hat{\mathbb{E}}[X_t | \mathcal{F}_{t-1}] = \frac{1}{w} \sum_{i=t-w}^{t-1} X_i \]

where \( w \) is the window size.

### B. Glossary

- **Adapted Process**: A process \( X_t \) is adapted to filtration \( \mathcal{F}_t \) if \( X_t \) is \( \mathcal{F}_t \)-measurable
- **Filtration**: Increasing sequence of σ-algebras representing information available over time
- **Martingale**: Process where conditional expectation of future value equals current value
- **Predictable Process**: Process where \( X_t \) is determined by information up to time \( t-1 \)
- **Submartingale**: Process where expected future values are at least current value
- **Supermartingale**: Process where expected future values are at most current value

### C. Quick Reference Commands

```python
# Basic decomposition
decomposer = DoobDecomposer()
result = decomposer.fit_transform(data)

# Load TTM model
from transformers import TinyTimeMixerForPrediction
model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM")

# Generate forecast
forecast = model.generate(past_values=context, prediction_length=96)

# Evaluate
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintained by**: Ars Probabilitas Team  
**License**: MIT

For questions, issues, or contributions, please visit:
https://github.com/Ars-Probabilitas/scenario-generation
