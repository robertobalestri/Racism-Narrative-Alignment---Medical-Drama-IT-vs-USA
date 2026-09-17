"""Per-series distribution of individual evaluations, with the series median marked."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import figure_path, load_scores

sns.set_theme(style="whitegrid")


def series_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Median score and first/last year of each series, in chronological order."""
    summary = (
        frame.groupby("series", observed=False)
        .agg(
            StartYear=("Year", "min"),
            EndYear=("Year", "max"),
            MedianScore=("score", "median"),
        )
        .reset_index()
        .sort_values(by="StartYear")
    )
    summary["Label"] = summary.apply(
        lambda row: f"{row['series']} ({int(row['StartYear'])}-{int(row['EndYear'])})",
        axis=1,
    )
    return summary


def create_distribution_plot(frame: pd.DataFrame, title: str, filename: str) -> None:
    summary = series_summary(frame)
    frame = frame.copy()
    frame["Label"] = frame["series"].map(dict(zip(summary["series"], summary["Label"])))
    order = summary["Label"].tolist()

    plt.figure(figsize=(14, 8))
    sns.stripplot(
        data=frame,
        x="Label",
        y="score",
        order=order,
        color="steelblue",
        alpha=0.6,
        size=4,
        jitter=0.25,
        zorder=1,
    )
    plt.scatter(
        np.arange(len(order)),
        summary["MedianScore"].values,
        color="crimson",
        marker="D",
        s=50,
        zorder=2,
    )
    plt.scatter([], [], color="gray", alpha=0.8, s=20, label="Single Evaluation")
    plt.scatter([], [], color="crimson", marker="D", s=40, label="Series Median")
    plt.axhline(0, color="gray", linestyle="dashed", linewidth=1, zorder=0)

    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.ylim(-3.5, 3.5)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.xlabel("TV Series Title (Start-End based on valid episodes)", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    output = figure_path(filename)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output}")


create_distribution_plot(
    load_scores("IT"),
    "Distribution and Median of Narrative Scores by TV Series - Italian Series (IT)",
    "racism_distribution_IT_EN.png",
)
create_distribution_plot(
    load_scores("US"),
    "Distribution and Median of Narrative Scores by TV Series - US Series (US)",
    "racism_distribution_US_EN.png",
)
