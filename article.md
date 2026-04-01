# Convergent Cross Mapping for Causal Inference in Public Policy Causal inference shapes policy decisions. Governments need it to predict
the effects of tax changes, environmental rules, and healthcare...

::::### Convergent Cross Mapping for Causal Inference in Public Policy 

Causal inference shapes policy decisions. Governments need it to predict
the effects of tax changes, environmental rules, and healthcare laws.
Traditional methods assume a fixed relationship between variables.
Public policy does not work that way. Policies and outcomes evolve
together. Feedback loops drive economic and social systems. Convergent
Cross Mapping (CCM) detects causality in these complex environments. It
reconstructs system dynamics to test whether one variable contains
information about another.

### Why Traditional Methods Fail
Standard causal models break down when policies and outcomes influence
each other. A carbon tax affects emissions, but public pressure from
rising emissions can lead to tax increases. Traditional methods assume a
clean separation between cause and effect. CCM does not. It tests
whether changes in one variable leave a trace in another. If they do, a
causal relationship exists.

CCM comes from nonlinear time series analysis. It uses state-space
reconstruction to test whether one variable predicts another. If
historical values of X improve the prediction of Y, then X influences Y.
If the reverse does not hold, the effect runs in one direction. If both
variables predict each other, feedback exists. CCM strengthens with more
data. A genuine causal relationship improves prediction as the dataset
grows.

### Public Policy Example: Electricity Prices and Renewable Subsidies
Governments subsidize renewable energy, expecting lower electricity
prices. But electricity prices also influence subsidies. Traditional
regressions struggle with this loop. CCM reveals the direction of
influence. If past subsidies predict price drops but past prices fail to
predict subsidies, subsidies drive prices. If both cross-map, feedback
exists.

This Python example uses CCM to analyze the relationship between
subsidies and prices.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def state_space_reconstruction(series, dim, tau):
    """Reconstructs the state space using time-delay embedding."""
    n = len(series) - (dim - 1) * tau
    return np.array([series[i:i + dim * tau:tau] for i in range(n)])

def cross_map(source, target, dim=3, tau=1):
    """Tests whether the source variable encodes information about the target."""
    source_embedded = state_space_reconstruction(source, dim, tau)
    target_embedded = target[:len(source_embedded)]  # Align lengths

    nn = NearestNeighbors(n_neighbors=dim).fit(source_embedded)
    _, indices = nn.kneighbors(source_embedded)

    predicted_target = np.mean(target[indices], axis=1)
    return np.corrcoef(target_embedded, predicted_target)[0, 1]

np.random.seed(42)
subsidies = np.cumsum(np.random.randn(200))  # Simulated subsidy data
prices = np.cumsum(np.random.randn(200))  # Simulated electricity price data

subsidy_predicts_price = cross_map(subsidies, prices)
price_predicts_subsidy = cross_map(prices, subsidies)

print(f"Subsidies predict electricity prices: {subsidy_predicts_price:.3f}")
print(f"Electricity prices predict subsidies: {price_predicts_subsidy:.3f}")
```

This test reveals whether subsidies drive prices or whether both evolve
together. If one direction shows a strong correlation while the other
does not, the causal influence runs one way.

### Public Health Example: Cigarette Taxes and Smoking Rates
Governments raise cigarette taxes to reduce smoking. But public
sentiment about smoking can lead to higher taxes. Traditional methods
cannot separate these effects. CCM does. It tests whether past tax
levels predict smoking declines or if smoking rates predict tax hikes.
If tax levels cross-map onto smoking but not the reverse, taxes drive
behavior. If both cross-map, policy and public opinion form a feedback
loop.
::::CCM does not assume a fixed relationship. It detects causality in
dynamic systems. It works when feedback loops distort standard models.
It handles nonlinearity without predefined equations. But it needs long
time series. It requires heavy computation. It identifies causality but
does not measure effect size.

Public policy rarely follows a simple cause-and-effect model. Policies
and outcomes shape each other over time. Traditional methods miss these
interactions. CCM reconstructs system dynamics to reveal causal links.
It works where standard techniques fail. It provides a tool for
understanding complex policy effects.
::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[March 1, 2025](https://medium.com/p/f4ab090cbe6f).

[Canonical
link](https://medium.com/@kyle-t-jones/convergent-cross-mapping-for-causal-inference-in-public-policy-f4ab090cbe6f)

Exported from [Medium](https://medium.com) on November 10, 2025.
