"""Stage 3: attach the original air date of each episode to the Likert results.

Air dates are curated by hand (data/airdates/airdates_*.csv) because they are not
part of the subtitle corpus. The merged file is what the analysis scripts read.

    python pipeline/03_merge_airdates.py --dataset IT
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import STUDY_FIRST_YEAR, STUDY_LAST_YEAR, dataset, out_of_period

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COLUMN_ORDER = [
    "series",
    "season",
    "episode",
    "episode_code",
    "data_messa_in_onda",
    "attempt_number",
    "score",
    "explanation",
    "why_not_lower",
    "why_not_higher",
]


def merge(dataset_name: str) -> None:
    paths = dataset(dataset_name)

    likert = pd.read_csv(paths["likert_csv"], dtype=str, keep_default_na=False)
    likert = likert[~likert["episode_code"].isin(out_of_period(dataset_name))]
    airdates = pd.read_csv(paths["airdates_csv"], dtype=str, keep_default_na=False)

    # The Likert file already carries series/season/episode; take only the date.
    airdates = airdates[["episode_code", "data_messa_in_onda"]]
    merged = likert.merge(airdates, on="episode_code", how="left")
    merged["data_messa_in_onda"] = merged["data_messa_in_onda"].fillna("")

    missing = merged["data_messa_in_onda"].eq("").sum()
    if missing:
        codes = sorted(set(merged.loc[merged["data_messa_in_onda"].eq(""), "episode_code"]))
        logger.warning(f"{missing} rows without an air date ({len(codes)} episodes): {codes}")

    years = pd.to_datetime(merged["data_messa_in_onda"], dayfirst=True, errors="coerce").dt.year
    outside = merged.loc[~years.between(STUDY_FIRST_YEAR, STUDY_LAST_YEAR) & years.notna(), "episode_code"]
    if not outside.empty:
        logger.warning(
            f"Episodes dated outside {STUDY_FIRST_YEAR}-{STUDY_LAST_YEAR}, "
            f"add them to {paths['out_of_period_csv'].name}: {sorted(set(outside))}"
        )

    output_csv = paths["final_csv"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig and ';' so the file opens correctly in an Italian Excel locale.
    merged[COLUMN_ORDER].to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")

    logger.info(f"{len(merged)} rows written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["IT", "US"])
    merge(parser.parse_args().dataset)
