"""Shared loading helpers for the analysis scripts."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, dataset

COUNTRY_LABELS = {"IT": "Italian Series", "US": "US Series"}


def load_scores(dataset_name: str) -> pd.DataFrame:
    """Loads the merged results of one dataset, keeping only scored, dated episodes.

    Evaluations answered "N/A" (the topic is absent from the episode) become NaN
    and are dropped here, exactly as in the published analysis.
    """
    paths = dataset(dataset_name)
    frame = pd.read_csv(paths["final_csv"], sep=";", encoding="utf-8-sig")
    frame["Air Date"] = pd.to_datetime(
        frame["data_messa_in_onda"], dayfirst=True, errors="coerce"
    )
    frame = frame.dropna(subset=["Air Date", "score"]).copy()
    frame["Year"] = frame["Air Date"].dt.year
    frame["Country"] = COUNTRY_LABELS[dataset_name.upper()]
    return frame


def figure_path(filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename
