"""Stability of the ten evaluations each episode receives.

Reports how often the model repeats itself and how wide the spread is when it
does not. "N/A" answers (the topic is absent) are counted as a category of their
own, so an episode judged N/A ten times counts as unanimous rather than empty.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FINAL_DIR, dataset


def load_raw_scores(dataset_name: str) -> pd.DataFrame:
    """Loads the merged results without dropping the "N/A" evaluations."""
    frame = pd.read_csv(
        dataset(dataset_name)["final_csv"],
        sep=";",
        encoding="utf-8-sig",
        dtype={"score": str},
        keep_default_na=False,
    )
    frame["score_label"] = frame["score"].str.strip().str.upper().replace({"+1": "1", "+2": "2", "+3": "3"})
    frame["score_numeric"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame


def per_episode_stability(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("episode_code")
    numeric = frame.dropna(subset=["score_numeric"]).groupby("episode_code")["score_numeric"]

    stability = pd.DataFrame(
        {
            "n_evaluations": grouped.size(),
            "n_distinct_answers": grouped["score_label"].nunique(),
            "n_na": grouped["score_label"].apply(lambda values: (values == "N/A").sum()),
            "modal_answer": grouped["score_label"].agg(lambda values: values.mode().iat[0]),
        }
    )
    stability["median"] = numeric.median()
    stability["iqr"] = numeric.quantile(0.75) - numeric.quantile(0.25)
    stability["range"] = numeric.max() - numeric.min()
    stability["std"] = numeric.std()
    stability["unanimous"] = stability["n_distinct_answers"] == 1
    return stability.reset_index()


def report(dataset_name: str) -> pd.DataFrame:
    stability = per_episode_stability(load_raw_scores(dataset_name))
    episodes = len(stability)
    unanimous = int(stability["unanimous"].sum())
    mixed_na = int(((stability["n_na"] > 0) & (stability["n_na"] < stability["n_evaluations"])).sum())
    scored = stability.dropna(subset=["iqr"])

    print(f"=== {dataset_name}: {episodes} episodes x 10 evaluations ===")
    print(f"unanimous (all 10 identical): {unanimous}/{episodes} ({100 * unanimous / episodes:.1f}%)")
    print(f"episodes mixing N/A and a score: {mixed_na}/{episodes} ({100 * mixed_na / episodes:.1f}%)")
    print(f"mean IQR of numeric scores: {scored['iqr'].mean():.3f} (median {scored['iqr'].median():.3f})")
    print(f"share with IQR = 0: {100 * scored['iqr'].eq(0).mean():.1f}%")
    print(f"mean range (max-min): {scored['range'].mean():.3f}")
    print(f"mean standard deviation: {scored['std'].mean():.3f}")
    print()

    output = FINAL_DIR / f"stability_{dataset_name}.csv"
    stability.to_csv(output, index=False, encoding="utf-8")
    print(f"Per-episode stability written to {output}\n")
    return stability


if __name__ == "__main__":
    for name in ("IT", "US"):
        report(name)
