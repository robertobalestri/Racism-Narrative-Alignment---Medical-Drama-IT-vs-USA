# Diachronic Evolution of Narrative Alignment (linear regression)

## Final sample

| Dataset | Valid episodes / corpus | Share | Evaluations (numeric) |
|---|---|---|---|
| Italian Series | 50 / 249 | 20.1% | 490 |
| US Series | 222 / 1553 | 14.3% | 2081 |

Valid episode: at most 5 of its 10 evaluations are N/A.

## Linear trend of the score over air date

One point per valid episode: the median of its numeric evaluations, plotted
against its air date. Ordinary least squares on the air date as a decimal year
(`scipy.stats.linregress`); p tests slope = 0, two-sided. Slope is in score
points per decade, with its 95% confidence interval. The line in the figure
(seaborn `lmplot`) is the same regression on the same points; its band is
seaborn's bootstrapped 95% confidence interval of the fitted line.

| Dataset | Episodes | Slope per decade (95% CI) | R² | p |
|---|---|---|---|---|
| Italian Series | 50 | +0.193 (-0.157 to +0.543) | 0.025 | = .273 |
| US Series | 222 | +0.492 (+0.363 to +0.622) | 0.203 | < .001 |
