# sk-causal

Causal inference algorithms with scikit-learn compatible APIs.

## Installation

```bash
pip install sk-causal
```

Or install from source:

```bash
git clone https://github.com/your-username/sk-causal.git
cd sk-causal
pip install -e .
```

## Quick Start

```python
import numpy as np
from sk_causal import PropensityScoreMatching, InversePropensityWeighting, DoublyRobust

# Generate synthetic data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
treatment = (np.random.rand(n) < propensity).astype(int)
y = X[:, 0] + X[:, 1] + treatment * 2.0 + np.random.randn(n) * 0.5  # True ATE = 2.0

# Propensity Score Matching
psm = PropensityScoreMatching(n_neighbors=3)
psm.fit(X, treatment, y)
print(f"PSM ATE estimate: {psm.estimate_ate():.3f}")

# Inverse Propensity Weighting
ipw = InversePropensityWeighting()
ipw.fit(X, treatment, y)
print(f"IPW ATE estimate: {ipw.estimate_ate():.3f}")

# Doubly Robust (AIPW)
dr = DoublyRobust()
dr.fit(X, treatment, y)
print(f"DR ATE estimate: {dr.estimate_ate():.3f}")
```

## Estimators

### PropensityScoreMatching

Estimates treatment effects by matching treated units to control units with similar propensity scores.

```python
from sk_causal import PropensityScoreMatching

psm = PropensityScoreMatching(
    n_neighbors=1,      # Number of neighbors for matching
    caliper=None,       # Maximum distance for valid matches
    propensity_model=None,  # Custom propensity model (default: LogisticRegression)
    random_state=42
)
psm.fit(X, treatment, y)

ate = psm.estimate_ate()  # Average Treatment Effect
att = psm.estimate_att()  # Average Treatment Effect on the Treated
```

### InversePropensityWeighting

Uses inverse probability of treatment weighting to estimate treatment effects.

```python
from sk_causal import InversePropensityWeighting

ipw = InversePropensityWeighting(
    propensity_model=None,      # Custom propensity model
    clip=(0.01, 0.99),          # Clip propensity scores to avoid extreme weights
    normalize_weights=True,      # Use normalized (Hajek) estimator
    random_state=42
)
ipw.fit(X, treatment, y)

ate = ipw.estimate_ate()
```

### DoublyRobust

Combines propensity weighting with outcome regression. Consistent if either model is correctly specified.

```python
from sk_causal import DoublyRobust

dr = DoublyRobust(
    propensity_model=None,  # Custom propensity model
    outcome_model=None,     # Custom outcome model (default: Ridge)
    clip=(0.01, 0.99),      # Clip propensity scores
    random_state=42
)
dr.fit(X, treatment, y)

ate = dr.estimate_ate()
cate = dr.estimate_cate(X_new)  # Conditional ATE for new observations
```

## Using Custom Models

All estimators accept custom scikit-learn compatible models:

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sk_causal import DoublyRobust

dr = DoublyRobust(
    propensity_model=GradientBoostingClassifier(n_estimators=100),
    outcome_model=GradientBoostingRegressor(n_estimators=100)
)
dr.fit(X, treatment, y)
```

## API Reference

All estimators follow scikit-learn conventions:

- `fit(X, treatment, y)`: Fit the estimator
- `estimate_ate()`: Estimate Average Treatment Effect
- `estimate_att()`: Estimate Average Treatment Effect on the Treated (where available)
- `get_params()` / `set_params()`: Get/set parameters

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT License
