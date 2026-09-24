"""Spread of the per-episode median scores within each series, IT and US.

Each valid episode's evaluations are reduced to their median (N/A excluded); the
standard deviation of those medians says how uniformly a series treats the topic.
Only valid episodes (at most MAX_NA_EVALUATIONS "N/A" out of 10) are counted.
Writes results/final/series_dispersion.md.
"""

import pandas as pd

from common import decimal_it, load_scores
from src.config import FINAL_DIR, MAX_NA_EVALUATIONS

TITLES = {"IT": "Serie italiane", "US": "Serie statunitensi"}


def series_dispersion(frame: pd.DataFrame) -> pd.DataFrame:
    medians = (
        frame.dropna(subset=["score"])
        .groupby(["series", "episode_code"])["score"]
        .median()
    )
    return (
        medians.groupby(level="series")
        .agg(
            episodes="size",
            std_population=lambda values: values.std(ddof=0),
            std_sample="std",
            minimum="min",
            maximum="max",
            medians=lambda values: values.value_counts().sort_index().to_dict(),
        )
        .sort_values(["std_population", "episodes"], ascending=[True, False])
    )


def markdown_table(summary: pd.DataFrame) -> list:
    lines = [
        "| Serie | Episodi | DS popolazione | DS campionaria | Min-max | Mediane (valore x episodi) |",
        "|---|---|---|---|---|---|",
    ]
    for series, row in summary.iterrows():
        sample = "n.d." if pd.isna(row["std_sample"]) else decimal_it(row["std_sample"], 2)
        medians = "; ".join(
            f"{decimal_it(value, 1)} x{count}" for value, count in row["medians"].items()
        )
        lines.append(
            f"| {series} | {row['episodes']} | {decimal_it(row['std_population'], 2)} "
            f"| {sample} | {decimal_it(row['minimum'], 1)} - {decimal_it(row['maximum'], 1)} "
            f"| {medians} |"
        )
    return lines


if __name__ == "__main__":
    lines = [
        "# Dispersione delle mediane per serie",
        "",
        f"Solo episodi validi (al massimo {MAX_NA_EVALUATIONS} N/A su 10). Per ogni episodio",
        "si prende la mediana delle valutazioni numeriche, poi si calcola la deviazione",
        "standard (DS) di queste mediane all'interno della serie.",
        "La DS campionaria (ddof=1) non e' definita per le serie con un solo episodio.",
    ]
    for name in ("IT", "US"):
        summary = series_dispersion(load_scores(name))
        zero = summary.index[summary["std_population"].eq(0)].tolist()
        lines += ["", f"## {TITLES[name]}", ""]
        lines += markdown_table(summary)
        lines += ["", f"Serie con DS nulla: {', '.join(zero) if zero else 'nessuna'}."]

    output = FINAL_DIR / "series_dispersion.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output}")
