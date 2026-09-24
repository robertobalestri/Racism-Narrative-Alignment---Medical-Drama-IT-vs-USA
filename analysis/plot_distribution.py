"""Per-series distribution of episode scores, with the series median marked.

One point per valid episode (the median of its numeric evaluations), as in the
other figures: an episode counts once, whatever its number of evaluations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import SERIES_RUNS, episode_scores, figure_path, load_scores, write_notes
from series_dispersion import series_dispersion

sns.set_theme(style="whitegrid")


def series_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Median of the episode scores of each series, ordered by broadcast run.

    The label carries the series' original run (SERIES_RUNS); StartYear and EndYear
    are the first and last air year of its valid episodes, reported in the notes.
    """
    summary = (
        episodes.assign(Year=episodes["air_date"].dt.year)
        .groupby("series", observed=False)
        .agg(
            StartYear=("Year", "min"),
            EndYear=("Year", "max"),
            MedianScore=("score", "median"),
        )
        .reset_index()
    )
    summary["Run"] = summary["series"].map(SERIES_RUNS)
    summary = summary.sort_values(by=["Run", "series"])
    summary["Label"] = summary["series"] + " (" + summary["Run"] + ")"
    return summary


def create_distribution_plot(frame: pd.DataFrame, title: str, filename: str) -> None:
    episodes = episode_scores(frame)
    summary = series_summary(episodes)
    episodes["Label"] = episodes["series"].map(dict(zip(summary["series"], summary["Label"])))
    order = summary["Label"].tolist()

    plt.figure(figsize=(14, 8))
    sns.stripplot(
        data=episodes,
        x="Label",
        y="score",
        order=order,
        color="steelblue",
        alpha=0.6,
        size=6,
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
    plt.scatter([], [], color="gray", alpha=0.8, s=20, label="Episode (median of its evaluations)")
    plt.scatter([], [], color="crimson", marker="D", s=40, label="Series Median")
    plt.axhline(0, color="gray", linestyle="dashed", linewidth=1, zorder=0)

    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    plt.ylim(-3.5, 3.5)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.xlabel("TV Series Title (Original Broadcast Run)", fontsize=11)
    plt.ylabel("Score (episode median)", fontsize=11)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    output = figure_path(filename)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output}")
    write_notes(filename, distribution_notes(frame, summary, title))


def distribution_notes(frame: pd.DataFrame, summary: pd.DataFrame, title: str) -> list:
    spread = series_dispersion(frame)
    evaluations = frame.groupby("series")["score"].size()
    lines = [
        f"# {title}",
        "",
        "Valid episodes only (at most 5 N/A out of 10), numeric evaluations only.",
        "",
        "- Dots: one per valid episode, at the median of its numeric evaluations,",
        "  jittered horizontally.",
        "- Red diamond (series median): median of the episode medians of the series.",
        "  Each episode weighs once, however many numeric evaluations it has.",
        "- Label years: the original broadcast run of the series, as stated in the",
        "  paper. Valid episodes: first and last air year of the valid episodes.",
        "- SD of episode medians: each episode is first reduced to the median of its",
        "  evaluations, then the standard deviation of those medians is taken within",
        "  the series. Population SD (ddof=0) and sample SD (ddof=1); the sample SD is",
        "  undefined (n/a) for a series with a single episode. An SD of 0 means every",
        "  episode of the series has the same median score.",
        "",
        "| Series | Run | Valid episodes aired | Episodes | Evaluations | Series median | SD of episode medians (pop. / sample) | Range of episode medians |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        series = row["series"]
        sd = spread.loc[series]
        sample_sd = "n/a" if pd.isna(sd["std_sample"]) else f"{sd['std_sample']:.2f}"
        lines.append(
            f"| {series} | {row['Run']} | {int(row['StartYear'])}-{int(row['EndYear'])} | {sd['episodes']} "
            f"| {evaluations[series]} | {row['MedianScore']:+.2f} "
            f"| {sd['std_population']:.2f} / {sample_sd} "
            f"| {sd['minimum']:+.1f} to {sd['maximum']:+.1f} |"
        )
    zero = spread.index[spread["std_population"].eq(0)].tolist()
    lines += ["", f"Series with SD of episode medians = 0: {', '.join(zero) if zero else 'none'}."]
    return lines


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
