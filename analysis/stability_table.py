"""Stability indicators of the ten evaluations per episode, IT vs US.

Definitions:
- unanimous: all 10 evaluations are identical, N/A counting as an answer of its own,
  so an episode judged N/A ten times is unanimous (as in stability_stats.py). The
  share is over every episode that passed screening.
- valid episodes: at most MAX_NA_EVALUATIONS "N/A" out of 10, the final sample of
  the study. IQR, range and standard deviation are computed on their numeric
  scores only, and their shares and means are over valid episodes.
Writes results/final/stability_table.md.
"""

import pandas as pd

from common import decimal_it, load_evaluations
from src.config import FINAL_DIR, MAX_NA_EVALUATIONS, NUMBER_OF_ATTEMPTS


def indicators(frame: pd.DataFrame) -> dict:
    scores = frame.groupby("episode_code")["score"]
    episodes = scores.ngroups
    unanimous = int(
        ((scores.size() == NUMBER_OF_ATTEMPTS) & (scores.nunique(dropna=False) == 1)).sum()
    )
    na_count = scores.size() - scores.count()
    valid = na_count <= MAX_NA_EVALUATIONS
    iqr = (scores.quantile(0.75) - scores.quantile(0.25))[valid]
    spread = (scores.max() - scores.min())[valid]
    return {
        "episodes": episodes,
        "Episodi con valutazioni unanimi (10/10)": share(unanimous, episodes),
        f"Episodi validi (al massimo {MAX_NA_EVALUATIONS} N/A su 10)": share(int(valid.sum()), episodes),
        "Episodi validi con scarto interquartile nullo": share(int(iqr.eq(0).sum()), int(valid.sum())),
        "Scarto interquartile medio (mediana)": f"{decimal_it(iqr.mean())} ({decimal_it(iqr.median())})",
        "Intervallo medio (massimo - minimo)": decimal_it(spread.mean()),
        "Deviazione standard media": decimal_it(scores.std()[valid].mean()),
    }


def share(count: int, total: int) -> str:
    return f"{count}/{total} ({decimal_it(100 * count / total, 1)}%)"


italy = indicators(load_evaluations("IT"))
usa = indicators(load_evaluations("US"))

lines = [
    "# Stabilita' delle valutazioni",
    "",
    f"| Indicatore | Italia ({italy['episodes']} episodi) | Stati Uniti ({usa['episodes']} episodi) |",
    "|---|---|---|",
]
lines += [f"| {key} | {italy[key]} | {usa[key]} |" for key in italy if key != "episodes"]

output = FINAL_DIR / "stability_table.md"
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nReport written to {output}")
