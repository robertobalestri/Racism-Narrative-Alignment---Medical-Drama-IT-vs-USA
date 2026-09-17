"""Stage 1: ask GPT-4.1 whether each episode touches on racism at all.

Only the episodes answered with "Yes" go on to the Likert evaluation in stage 2.

    python pipeline/01_screen_episodes.py --dataset IT
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SCREENING_PROMPT, dataset
from src.llm_client import GPT41Client
from src.srt_parser import convert_srts_to_json, load_subtitles_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIELDNAMES = ["series", "season", "episode", "episode_code", "bool_response", "explanation"]


def load_existing(csv_path: Path) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not csv_path.is_file():
        return [], set()

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return rows, {row["episode_code"] for row in rows if row.get("episode_code")}


def save(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def episode_metadata(json_path: Path) -> Dict[str, Any]:
    """Reads series, season and episode from the .../SERIES/S01/E01/CODE.json layout."""
    parts = json_path.parts
    return {
        "series": parts[-4],
        "season": int(parts[-3][1:]),
        "episode": int(parts[-2][1:]),
        "episode_code": json_path.stem,
    }


def screen(dataset_name: str) -> None:
    paths = dataset(dataset_name)
    subtitles_dir = paths["subtitles_dir"]
    output_csv = paths["screening_csv"]

    if not subtitles_dir.is_dir():
        logger.error(f"Subtitles directory not found: {subtitles_dir}")
        return

    base_prompt = SCREENING_PROMPT.read_text(encoding="utf-8")
    convert_srts_to_json(subtitles_dir)

    client = GPT41Client()
    rows, processed = load_existing(output_csv)
    logger.info(f"{len(processed)} episodes already screened.")

    for json_path in sorted(subtitles_dir.rglob("*.json")):
        try:
            metadata = episode_metadata(json_path)
        except (IndexError, ValueError) as error:
            logger.warning(f"Cannot read metadata from {json_path}: {error}")
            continue

        if metadata["episode_code"] in processed:
            continue

        subtitles_text = load_subtitles_text(json_path)
        if not subtitles_text.strip():
            logger.warning(f"No dialogue in {json_path.name}, skipped.")
            continue

        answer = client.complete_json(
            base_prompt + subtitles_text, metadata["episode_code"]
        )
        if answer is None:
            row = {**metadata, "bool_response": "Error", "explanation": "LLM call failed."}
        else:
            verdict = str(answer.get("answer", "")).strip().lower()
            row = {
                **metadata,
                "bool_response": verdict.capitalize() if verdict in ("yes", "no") else "Error",
                "explanation": str(answer.get("explanation", "")).strip() if verdict == "yes" else "",
            }

        logger.info(f"{metadata['episode_code']}: {row['bool_response']}")
        rows.append(row)
        processed.add(metadata["episode_code"])
        save(rows, output_csv)

    positives = sum(1 for row in rows if row.get("bool_response") == "Yes")
    logger.info(f"Done. {len(rows)} episodes screened, {positives} flagged as relevant.")
    logger.info(f"Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["IT", "US"])
    screen(parser.parse_args().dataset)
