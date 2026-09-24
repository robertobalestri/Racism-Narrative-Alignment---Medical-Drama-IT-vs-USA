"""Scatter plot with linear trend of alignment scores over air date, IT vs US.

One point per valid episode (the median of its numeric evaluations): the ten
evaluations of an episode are not independent observations, so the line, its
confidence band and the reported statistics are all estimated on episodes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from common import (
    COUNTRY_LABELS,
    corpus_size,
    episode_scores,
    figure_path,
    load_scores,
    p_value,
    set_year_ticks,
    write_notes,
)

sns.set_theme(style="whitegrid")

FIGURE = "racism_alignment_evolution_EN.png"

scores = {name: load_scores(name) for name in ("IT", "US")}
episodes = {name: episode_scores(frame) for name, frame in scores.items()}
combined = pd.concat(episodes.values(), ignore_index=True)
combined["Air Date Ordinal"] = combined["air_date"].map(pd.Timestamp.toordinal)

grid = sns.lmplot(
    data=combined,
    x="Air Date Ordinal",
    y="score",
    hue="Country",
    height=6,
    aspect=1.5,
    scatter_kws={"alpha": 0.6, "s": 40},
    legend_out=False,
)

axes = grid.ax
set_year_ticks(axes)
plt.xticks(rotation=45)

plt.xlabel("Air Date (Year)", fontsize=12)
plt.ylabel("Narrative Alignment Score (Racism), episode median", fontsize=12)
plt.title("Diachronic Evolution of Narrative Alignment", fontsize=14)
plt.legend(title="Dataset", title_fontsize="13", fontsize="11")
plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.ylim(-3.2, 3.2)
plt.tight_layout()

output = figure_path(FIGURE)
grid.savefig(output, dpi=300)
print(f"Plot saved to {output}")


def linear_trend(years: pd.Series, values: pd.Series) -> str:
    fit = stats.linregress(years, values)
    margin = stats.t.ppf(0.975, len(values) - 2) * fit.stderr
    return (
        f"| {len(values)} | {fit.slope * 10:+.3f} "
        f"({(fit.slope - margin) * 10:+.3f} to {(fit.slope + margin) * 10:+.3f}) "
        f"| {fit.rvalue ** 2:.3f} | {p_value(fit.pvalue)} |"
    )


lines = [
    "# Diachronic Evolution of Narrative Alignment (linear regression)",
    "",
    "## Final sample",
    "",
    "| Dataset | Valid episodes / corpus | Share | Evaluations (numeric) |",
    "|---|---|---|---|",
]
for name, frame in scores.items():
    valid, corpus = len(episodes[name]), corpus_size(name)
    lines.append(
        f"| {COUNTRY_LABELS[name]} | {valid} / {corpus} | {100 * valid / corpus:.1f}% | {len(frame)} |"
    )
lines += [
    "",
    "Valid episode: at most 5 of its 10 evaluations are N/A.",
    "",
    "## Linear trend of the score over air date",
    "",
    "One point per valid episode: the median of its numeric evaluations, plotted",
    "against its air date. Ordinary least squares on the air date as a decimal year",
    "(`scipy.stats.linregress`); p tests slope = 0, two-sided. Slope is in score",
    "points per decade, with its 95% confidence interval. The line in the figure",
    "(seaborn `lmplot`) is the same regression on the same points; its band is",
    "seaborn's bootstrapped 95% confidence interval of the fitted line.",
    "",
    "| Dataset | Episodes | Slope per decade (95% CI) | R² | p |",
    "|---|---|---|---|---|",
]
for name, frame in episodes.items():
    lines.append(f"| {COUNTRY_LABELS[name]} " + linear_trend(frame["decimal_year"], frame["score"]))
write_notes(FIGURE, lines)
