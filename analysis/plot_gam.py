"""GAM trend of alignment scores over air date, with 95% confidence bands.

One point per valid episode (the median of its numeric evaluations): the ten
evaluations of an episode are not independent observations, so the curves, their
bands and the reported statistics are all estimated on episodes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pygam import LinearGAM, s

from common import (
    COUNTRY_LABELS,
    episode_scores,
    figure_path,
    load_scores,
    p_value,
    set_year_ticks,
    write_notes,
)

FIGURE = "racism_alignment_gam_EN.png"

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
colors = {
    "Italian Series": sns.color_palette()[0],
    "US Series": sns.color_palette()[1],
}

rows = []
for name in ("IT", "US"):
    episodes = episode_scores(load_scores(name))
    country = COUNTRY_LABELS[name]
    color = colors[country]
    features = episodes[["air_date"]].map(pd.Timestamp.toordinal).values
    scores = episodes["score"].values

    plt.scatter(features, scores, alpha=0.5, s=30, color=color, label=f"{country} (Episodes)")

    gam = LinearGAM(s(0)).fit(features, scores)
    grid = gam.generate_X_grid(term=0, n=200)
    plt.plot(grid, gam.predict(grid), color=color, linewidth=2, label=f"{country} (GAM Trend)")

    bands = gam.confidence_intervals(grid, width=0.95)
    plt.fill_between(grid[:, 0], bands[:, 0], bands[:, 1], color=color, alpha=0.2)

    statistics = gam.statistics_
    rows.append(
        f"| {country} | {len(scores)} | {statistics['edof']:.2f} "
        f"| {statistics['pseudo_r2']['explained_deviance']:.3f} "
        f"| {p_value(statistics['p_values'][0])} |"
    )

axes = plt.gca()
set_year_ticks(axes)
plt.xticks(rotation=45)

plt.xlabel("Air Date (Year)", fontsize=12)
plt.ylabel("Narrative Alignment Score (Racism), episode median", fontsize=12)
plt.title("GAM Analysis: Diachronic Evolution of Narrative Alignment", fontsize=14)
plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.ylim(-3.2, 3.2)
plt.legend(
    title="Dataset and Trend",
    title_fontsize="13",
    fontsize="11",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.tight_layout()

output = figure_path(FIGURE)
plt.savefig(output, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output}")

write_notes(
    FIGURE,
    [
        "# GAM Analysis: Diachronic Evolution of Narrative Alignment",
        "",
        "One point per valid episode (at most 5 N/A out of 10): the median of its",
        "numeric evaluations, plotted against its air date.",
        "",
        "Generalised additive model `score ~ s(air date)` fitted separately per dataset",
        "with pygam `LinearGAM(s(0))`: one penalised cubic spline, 20 basis functions,",
        "default smoothing (lam = 0.6), Gaussian errors. The p value is pygam's test",
        "that the smooth term is zero, i.e. that air date has no effect; pygam notes",
        "that these p values are approximate. EDoF is the effective degrees of freedom",
        "of the model: the higher, the more the curve bends. Pseudo R² is the",
        "explained deviance. The curves and 95% bands in the figure come from these",
        "same models.",
        "",
        "| Dataset | Episodes | EDoF | Pseudo R² | p |",
        "|---|---|---|---|---|",
        *rows,
    ],
)
