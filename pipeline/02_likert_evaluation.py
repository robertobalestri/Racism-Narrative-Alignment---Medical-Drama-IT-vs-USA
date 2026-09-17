"""Stage 2: score every relevant episode on the -3..+3 narrative alignment scale.

Each episode is evaluated NUMBER_OF_ATTEMPTS times so that the stability of the
judgement can be measured afterwards. Results are appended to the CSV after each
episode, so an interrupted run can simply be restarted.

    python pipeline/02_likert_evaluation.py --dataset IT
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NUMBER_OF_ATTEMPTS, dataset
from src.llm_client import GPT41Client
from src.srt_parser import convert_srts_to_json, load_subtitles_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "series",
    "season",
    "episode",
    "episode_code",
    "attempt_number",
    "score",
    "explanation",
    "why_not_lower",
    "why_not_higher",
]


def parse_episode_code(code: str) -> Optional[Dict[str, str]]:
    """Splits 'Medicina generaleS01E10' into series, season and episode."""
    match = re.match(r"^(.*)S(\d{2})E(\d{2})$", code, re.IGNORECASE)
    if not match:
        return None
    return {
        "series": match.group(1).strip().upper(),
        "season": match.group(2),
        "episode": match.group(3),
    }


def load_relevant_episodes(screening_csv: Path) -> Set[str]:
    """Returns the episode codes the screening stage answered 'Yes' for."""
    with open(screening_csv, "r", newline="", encoding="utf-8") as handle:
        return {
            row["episode_code"].strip()
            for row in csv.DictReader(handle)
            if row.get("bool_response", "").strip() == "Yes" and row.get("episode_code")
        }


def load_existing(csv_path: Path) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, int]]]:
    if not csv_path.is_file():
        return [], set()

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    done = {
        (row["episode_code"], int(row["attempt_number"]))
        for row in rows
        if row.get("episode_code") and str(row.get("attempt_number", "")).isdigit()
    }
    return rows, done


def save(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDNAMES} for row in rows)


def evaluate(dataset_name: str) -> None:
    paths = dataset(dataset_name)
    subtitles_dir = paths["subtitles_dir"]
    output_csv = paths["likert_csv"]

    if not paths["screening_csv"].is_file():
        logger.error(f"Screening results not found: {paths['screening_csv']}. Run stage 1 first.")
        return

    base_prompt = paths["likert_prompt"].read_text(encoding="utf-8")
    relevant = load_relevant_episodes(paths["screening_csv"])
    logger.info(f"{len(relevant)} episodes to evaluate.")

    convert_srts_to_json(subtitles_dir)
    client = GPT41Client()
    rows, done = load_existing(output_csv)

    for episode_code in sorted(relevant):
        metadata = parse_episode_code(episode_code)
        if metadata is None:
            logger.warning(f"Cannot parse episode code '{episode_code}', skipped.")
            continue

        pending = [n for n in range(1, NUMBER_OF_ATTEMPTS + 1) if (episode_code, n) not in done]
        if not pending:
            continue

        json_paths = list(subtitles_dir.rglob(f"{episode_code}.json"))
        if not json_paths:
            logger.warning(f"No subtitles found for {episode_code}, skipped.")
            continue

        subtitles_text = load_subtitles_text(json_paths[0])
        if not subtitles_text.strip():
            logger.warning(f"No dialogue for {episode_code}, skipped.")
            continue

        prompt = base_prompt.replace("{subtitles_text}", subtitles_text)

        for attempt_number in pending:
            label = f"{episode_code}-attempt{attempt_number}"
            answer = client.complete_json(prompt, label) or {
                "score": "LLM_CALL_ERROR",
                "explanation": "LLM call failed after retries.",
            }
            rows.append(
                {
                    **metadata,
                    "episode_code": episode_code,
                    "attempt_number": attempt_number,
                    "score": answer.get("score", "LLM_RESPONSE_KEY_ERROR"),
                    "explanation": answer.get("explanation", ""),
                    "why_not_lower": answer.get("why_not_lower", ""),
                    "why_not_higher": answer.get("why_not_higher", ""),
                }
            )
            done.add((episode_code, attempt_number))
            logger.info(f"{label}: score={rows[-1]['score']}")

        save(rows, output_csv)

    logger.info(f"Done. {len(rows)} evaluations written to {output_csv}")


def demo() -> None:
    assert parse_episode_code("Medicina generaleS01E10") == {
        "series": "MEDICINA GENERALE",
        "season": "01",
        "episode": "10",
    }
    assert parse_episode_code("CMS02E09")["series"] == "CM"
    assert parse_episode_code("Stargate SG1 S05E10")["series"] == "STARGATE SG1"
    assert parse_episode_code("not an episode") is None
    print("02_likert_evaluation demo OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["IT", "US"])
    parser.add_argument("--demo", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dataset:
        evaluate(args.dataset)
    else:
        parser.error("--dataset is required unless --demo is given")
