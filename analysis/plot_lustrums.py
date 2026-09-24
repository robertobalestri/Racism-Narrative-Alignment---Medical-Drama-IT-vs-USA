"""Mean alignment score per five-year period: one chart per dataset, plus a combined one."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import figure_path, load_scores, write_notes

sns.set_theme(style="whitegrid")

# Bins are right-closed on the air year. The first one starts at 1993 so that it
# takes in 1994, the first year of the study period: that period is six years long.
LUSTRUM_BINS = [1993, 1999, 2004, 2009, 2014, 2019, 2024]
LUSTRUM_LABELS = [
    "1994-1999",
    "2000-2004",
    "2005-2009",
    "2010-2014",
    "2015-2019",
    "2020-2024",
]


def lustrum_means(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Lustrum"] = pd.cut(frame["Year"], bins=LUSTRUM_BINS, labels=LUSTRUM_LABELS)
    return (
        frame.groupby("Lustrum", observed=False)
        .agg(
            score=("score", "mean"),
            sd=("score", "std"),
            episodes=("episode_code", "nunique"),
            evaluations=("score", "size"),
        )
        .reset_index()
    )


NOTES_HEADER = [
    "Bar height: mean of all numeric evaluations of the valid episodes (at most 5",
    "N/A out of 10) first broadcast in that five-year period; each episode weighs as",
    "many times as it has numeric evaluations. Lustrums are by air year, e.g.",
    "2010-2014 covers 1 January 2010 to 31 December 2014; the first period,",
    "1994-1999, is six years long so that 1994 is included. SD is the standard",
    "deviation of those evaluations. A missing bar means no valid episode aired in",
    "that period, not a mean of zero.",
]


def lustrum_table(means: pd.DataFrame) -> list:
    lines = ["| Lustrum | Episodes | Evaluations | Mean | SD |", "|---|---|---|---|---|"]
    for _, row in means.iterrows():
        if row["episodes"] == 0:
            lines.append(f"| {row['Lustrum']} | 0 | 0 | - | - |")
            continue
        sd = "n/a" if pd.isna(row["sd"]) else f"{row['sd']:.2f}"
        lines.append(
            f"| {row['Lustrum']} | {row['episodes']} | {row['evaluations']} | {row['score']:+.2f} | {sd} |"
        )
    return lines


def annotate_bars(axes) -> None:
    for patch in axes.patches:
        height = patch.get_height()
        if np.isnan(height) or abs(height) < 1e-9:
            continue
        axes.annotate(
            f"{height:.2f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            xytext=(0, 5 if height >= 0 else -5),
            textcoords="offset points",
            fontsize=9,
        )


def style_axes(title: str) -> None:
    plt.axhline(0, color="gray", linestyle="dashed", linewidth=1)
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.ylim(-3.2, 3.2)
    plt.xticks(rotation=45)
    plt.xlabel("Period (Lustrum)", fontsize=11)
    plt.ylabel("Mean of Scores", fontsize=11)
    plt.title(title, fontsize=13, pad=15)
    plt.tight_layout()


def single_dataset_plot(means: pd.DataFrame, title: str, color, filename: str) -> None:
    plt.figure(figsize=(10, 6))
    axes = sns.barplot(data=means, x="Lustrum", y="score", color=color, edgecolor="white")
    annotate_bars(axes)
    style_axes(title)

    output = figure_path(filename)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output}")
    write_notes(filename, [f"# {title}", "", *NOTES_HEADER, "", *lustrum_table(means)])


def combined_plot(means_it: pd.DataFrame, means_us: pd.DataFrame, filename: str) -> None:
    means_it, means_us = means_it.assign(Country="IT"), means_us.assign(Country="US")
    combined = pd.concat([means_it, means_us], ignore_index=True)

    plt.figure(figsize=(12, 7))
    axes = sns.barplot(
        data=combined,
        x="Lustrum",
        y="score",
        hue="Country",
        palette={
            "IT": sns.color_palette("muted")[0],
            "US": sns.color_palette("deep")[1],
        },
        edgecolor="white",
    )
    annotate_bars(axes)
    style_axes("Mean Narrative Score per Lustrum - IT vs US")
    plt.legend(title="Country")

    output = figure_path(filename)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output}")
    write_notes(
        filename,
        [
            "# Mean Narrative Score per Lustrum - IT vs US",
            "",
            *NOTES_HEADER,
            "",
            "## Italian Series (IT)",
            "",
            *lustrum_table(means_it),
            "",
            "## US Series (US)",
            "",
            *lustrum_table(means_us),
        ],
    )


means_it = lustrum_means(load_scores("IT"))
means_us = lustrum_means(load_scores("US"))

single_dataset_plot(
    means_it,
    "Mean Narrative Score per Lustrum - Italian Series (IT)",
    sns.color_palette("deep")[2],
    "racism_lustrum_IT_EN.png",
)
single_dataset_plot(
    means_us,
    "Mean Narrative Score per Lustrum - US Series (US)",
    sns.color_palette("muted")[0],
    "racism_lustrum_US_EN.png",
)
combined_plot(means_it, means_us, "racism_lustrum_IT_US_EN.png")
