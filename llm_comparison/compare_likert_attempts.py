"""Internal stability of each LLM on the Likert test sample.

Every model evaluated the same episodes five times with the same prompt. For each
episode and model the five scores are compared with each other; the indicators
are the same ones used for the main corpus (analysis/stability_table.py), so the
two can be read side by side.

Layout read, next to this script:

    llm_comparison/<EPISODE>/<model>/attempt_<n>.json   ({"score": <int>, ...})

Models are compared only on the episodes that all of them evaluated, so every
model is measured on the same material. Writes likert_stability_by_episode.csv
and likert_stability_by_model.md next to this script.

    python llm_comparison/compare_likert_attempts.py
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DETAIL_CSV = DATA_DIR / "likert_stability_by_episode.csv"
SUMMARY_MD = DATA_DIR / "likert_stability_by_model.md"

# The models of the comparison.
MODELS = ["gpt-4.1", "gpt-4o-mini", "gemini-2.5", "deepseek-v3"]


def load_attempts() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("*/*/attempt_*.json")):
        score = json.loads(path.read_text(encoding="utf-8")).get("score")
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"Invalid score {score!r} in {path}")
        rows.append({"episode": path.parent.parent.name, "model": path.parent.name, "score": score})
    if not rows:
        raise ValueError(f"No attempt files found in {DATA_DIR}")
    return pd.DataFrame(rows)


def common_episodes(attempts: pd.DataFrame) -> list:
    coverage = attempts[attempts["model"].isin(MODELS)].groupby("episode")["model"].nunique()
    return sorted(coverage.index[coverage == len(MODELS)])


def per_episode(attempts: pd.DataFrame) -> pd.DataFrame:
    scores = attempts.groupby(["model", "episode"])["score"]
    return pd.DataFrame(
        {
            "attempts": scores.size(),
            "mean_score": scores.mean(),
            "unanimous": scores.nunique() == 1,
            "iqr": scores.quantile(0.75) - scores.quantile(0.25),
            "range": scores.max() - scores.min(),
            "std": scores.std(),
        }
    ).reset_index()


def per_model(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby("model")
        .agg(
            episodes=("episode", "size"),
            attempts=("attempts", "sum"),
            unanimous=("unanimous", "sum"),
            zero_iqr=("iqr", lambda values: int(values.eq(0).sum())),
            mean_iqr=("iqr", "mean"),
            mean_range=("range", "mean"),
            mean_std=("std", "mean"),
            mean_score=("mean_score", "mean"),
        )
        .reindex(MODELS)
    )


def markdown(summary: pd.DataFrame, episodes: list, excluded: list) -> list:
    # Higher is better for the two counts, lower for the three spreads.
    best = {
        "unanimous": summary["unanimous"].max(),
        "zero_iqr": summary["zero_iqr"].max(),
        "mean_iqr": summary["mean_iqr"].min(),
        "mean_range": summary["mean_range"].min(),
        "mean_std": summary["mean_std"].min(),
    }

    def cell(model: str, column: str, text: str) -> str:
        return f"**{text}**" if summary.loc[model, column] == best[column] else text

    total = len(episodes)
    lines = [
        "# Internal stability of the LLMs on the Likert test sample",
        "",
        f"{total} episodes, each evaluated 5 times by every model with the same prompt.",
        "Only the episodes evaluated by all four models are counted"
        + (f" (excluded: {', '.join(excluded)})." if excluded else "."),
        "",
        "Per episode and model, the five scores are compared with each other:",
        "",
        "- Unanimous: all five scores identical.",
        "- IQR = 0: the middle half of the five scores coincide.",
        "- IQR, range (max - min) and standard deviation (sample, ddof=1) are",
        "  averaged over the episodes. Lower means more stable.",
        "",
        "Best value per column in bold. Mean score is shown for context only: it is",
        "not a stability measure.",
        "",
        "| Model | Unanimous episodes | Episodes with IQR = 0 | Mean IQR | Mean range | Mean SD | Mean score |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, row in summary.iterrows():
        unanimous, zero_iqr = int(row.unanimous), int(row.zero_iqr)
        lines.append(
            f"| {model} "
            f"| {cell(model, 'unanimous', f'{unanimous}/{total} ({100 * unanimous / total:.0f}%)')} "
            f"| {cell(model, 'zero_iqr', f'{zero_iqr}/{total} ({100 * zero_iqr / total:.0f}%)')} "
            f"| {cell(model, 'mean_iqr', f'{row.mean_iqr:.2f}')} "
            f"| {cell(model, 'mean_range', f'{row.mean_range:.2f}')} "
            f"| {cell(model, 'mean_std', f'{row.mean_std:.3f}')} "
            f"| {row.mean_score:+.2f} |"
        )
    return lines


def main() -> None:
    attempts = load_attempts()
    episodes = common_episodes(attempts)
    excluded = sorted(set(attempts["episode"]) - set(episodes))
    compared = attempts[attempts["model"].isin(MODELS) & attempts["episode"].isin(episodes)]

    detail = per_episode(compared)
    detail.round(4).to_csv(DETAIL_CSV, index=False, encoding="utf-8")
    SUMMARY_MD.write_text(
        "\n".join(markdown(per_model(detail), episodes, excluded)) + "\n", encoding="utf-8"
    )
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
