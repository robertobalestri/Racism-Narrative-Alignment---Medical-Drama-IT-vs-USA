"""GAM trend of alignment scores over air date, with 95% confidence bands."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
from pygam import LinearGAM, s

from common import figure_path, load_scores

sns.set_theme(style="whitegrid")

combined = pd.concat([load_scores("IT"), load_scores("US")], ignore_index=True)
combined = combined.rename(columns={"score": "Racism Score"})
combined["Air Date Ordinal"] = combined["Air Date"].map(pd.Timestamp.toordinal)

plt.figure(figsize=(10, 6))
colors = {
    "Italian Series": sns.color_palette()[0],
    "US Series": sns.color_palette()[1],
}

for country, group in combined.groupby("Country"):
    color = colors[country]
    features = group[["Air Date Ordinal"]].values
    scores = group["Racism Score"].values

    plt.scatter(features, scores, alpha=0.4, s=30, color=color, label=f"{country} (Data)")

    gam = LinearGAM(s(0)).fit(features, scores)
    grid = gam.generate_X_grid(term=0, n=200)
    plt.plot(grid, gam.predict(grid), color=color, linewidth=2, label=f"{country} (GAM Trend)")

    bands = gam.confidence_intervals(grid, width=0.95)
    plt.fill_between(grid[:, 0], bands[:, 0], bands[:, 1], color=color, alpha=0.2)

axes = plt.gca()
year_ticks = [tick for tick in axes.get_xticks() if tick > 1000]
axes.xaxis.set_major_locator(FixedLocator(year_ticks))
axes.set_xticklabels(
    [pd.Timestamp.fromordinal(int(tick)).strftime("%Y") for tick in year_ticks]
)
plt.xticks(rotation=45)

plt.xlabel("Air Date (Year)", fontsize=12)
plt.ylabel("Narrative Alignment Score (Racism)", fontsize=12)
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

output = figure_path("racism_alignment_gam_EN.png")
plt.savefig(output, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output}")
