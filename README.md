# Convergent Cross Mapping for Causal Inference in Public Policy

Published: 2025-03-01
Medium: [https://medium.com/@kyle-t-jones/convergent-cross-mapping-for-causal-inference-in-public-policy-f4ab090cbe6f](https://medium.com/@kyle-t-jones/convergent-cross-mapping-for-causal-inference-in-public-policy-f4ab090cbe6f)

## Business context

Causal inference shapes policy decisions. Governments need it to predict the effects of tax changes, environmental rules, and healthcare laws. Traditional methods assume a fixed relationship between variables. Public policy does not work that way. Policies and outcomes evolve together. Feedback loops drive economic and social systems. Convergent Cross Mapping (CCM) detects causality in these complex environments. It reconstructs system dynamics to test whether one variable contains information about another.

Standard causal models break down when policies and outcomes influence each other. A carbon tax affects emissions, but public pressure from rising emissions can lead to tax increases. Traditional methods assume a clean separation between cause and effect. CCM does not. It tests whether changes in one variable leave a trace in another. If they do, a causal relationship exists.

CCM comes from nonlinear time series analysis. It uses state-space reconstruction to test whether one variable predicts another. If historical values of X improve the prediction of Y, then X influences Y. If the reverse does not hold, the effect runs in one direction. If both variables predict each other, feedback exists. CCM strengthens with more data. A genuine causal relationship improves prediction as the dataset grows.



## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).