# GAM Analysis: Diachronic Evolution of Narrative Alignment

One point per valid episode (at most 5 N/A out of 10): the median of its
numeric evaluations, plotted against its air date.

Generalised additive model `score ~ s(air date)` fitted separately per dataset
with pygam `LinearGAM(s(0))`: one penalised cubic spline, 20 basis functions,
default smoothing (lam = 0.6), Gaussian errors. The p value is pygam's test
that the smooth term is zero, i.e. that air date has no effect; pygam notes
that these p values are approximate. EDoF is the effective degrees of freedom
of the model: the higher, the more the curve bends. Pseudo R² is the
explained deviance. The curves and 95% bands in the figure come from these
same models.

| Dataset | Episodes | EDoF | Pseudo R² | p |
|---|---|---|---|---|
| Italian Series | 50 | 7.44 | 0.055 | = .905 |
| US Series | 222 | 11.08 | 0.292 | < .001 |
