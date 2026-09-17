"""Scatter plot with linear trend of alignment scores over air date, IT vs US."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator

from common import figure_path, load_scores

sns.set_theme(style="whitegrid")

combined = pd.concat([load_scores("IT"), load_scores("US")], ignore_index=True)
combined = combined.rename(columns={"score": "Racism Score"})
combined["Air Date Ordinal"] = combined["Air Date"].map(pd.Timestamp.toordinal)

grid = sns.lmplot(
    data=combined,
    x="Air Date Ordinal",
    y="Racism Score",
    hue="Country",
    height=6,
    aspect=1.5,
    scatter_kws={"alpha": 0.6, "s": 40},
    legend_out=False,
)

axes = grid.ax
year_ticks = [tick for tick in axes.get_xticks() if tick > 1000]
axes.xaxis.set_major_locator(FixedLocator(year_ticks))
axes.set_xticklabels(
    [pd.Timestamp.fromordinal(int(tick)).strftime("%Y") for tick in year_ticks]
)
plt.xticks(rotation=45)

plt.xlabel("Air Date (Year)", fontsize=12)
plt.ylabel("Narrative Alignment Score (Racism)", fontsize=12)
plt.title("Diachronic Evolution of Narrative Alignment", fontsize=14)
plt.legend(title="Dataset", title_fontsize="13", fontsize="11")
plt.yticks([-3, -2, -1, 0, 1, 2, 3])
plt.ylim(-3.2, 3.2)
plt.tight_layout()

output = figure_path("racism_alignment_evolution_EN.png")
grid.savefig(output, dpi=300)
print(f"Plot saved to {output}")
