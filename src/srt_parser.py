"""SRT parsing: reads subtitle files and turns them into dialogue line dictionaries."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Ordered by how often they appear in the corpus; the first one that decodes wins.
ENCODINGS_TO_TRY = [
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "cp1252",
    "latin-1",
    "utf-16-le",
    "utf-16-be",
    "gbk",
    "big5",
    "shift-jis",
    "euc-jp",
    "cp932",
    "euc-kr",
    "cp949",
]


def timecode_to_seconds(timecode: str) -> float:
    """Converts an SRT timecode (HH:MM:SS,mmm) to seconds."""
    if isinstance(timecode, (int, float)):
        return float(timecode)

    timecode = timecode.strip().replace(".", ",", 1)
    try:
        parsed = datetime.strptime(timecode, "%H:%M:%S,%f")
        return (
            parsed.hour * 3600
            + parsed.minute * 60
            + parsed.second
            + parsed.microsecond / 1_000_000
        )
    except ValueError:
        parts = timecode.split(":")
        if len(parts) != 3:
            raise ValueError(f"Cannot parse timecode format: {timecode}")
        seconds_parts = parts[2].split(",")
        seconds = float(seconds_parts[0])
        milliseconds = float("0." + seconds_parts[1]) if len(seconds_parts) > 1 else 0.0
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + seconds + milliseconds


def clean_srt_text(text: str) -> str:
    """Joins the lines of a subtitle block, strips markup tags and collapses whitespace."""
    lines = [line.strip() for line in text.splitlines()]
    cleaned = " ".join(line for line in lines if line)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _read_lines(filepath: Path) -> List[str]:
    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(filepath, "r", encoding=encoding) as handle:
                return handle.read().splitlines()
        except UnicodeError:
            continue
        except OSError as error:
            logger.error(f"Cannot access {filepath.name}: {error}")
            return []
    logger.error(f"Could not decode {filepath.name} with any known encoding.")
    return []


def _parse_block(buffer: List[str], filepath: Path) -> Dict[str, Any]:
    if len(buffer) < 3 or not buffer[0].strip().isdigit() or "-->" not in buffer[1]:
        if any(line.strip() for line in buffer):
            logger.warning(f"Skipping malformed block in {filepath.name}: {buffer}")
        return {}

    start_str, end_str = [part.strip() for part in buffer[1].split("-->")]
    return {
        "line_number": int(buffer[0].strip()),
        "start": timecode_to_seconds(start_str),
        "end": timecode_to_seconds(end_str),
        "text": clean_srt_text("\n".join(buffer[2:])),
    }


def parse_srt_file(filepath: Path) -> List[Dict[str, Any]]:
    """Parses an SRT file into a list of dialogue line dictionaries."""
    lines = _read_lines(filepath)
    if not lines:
        return []

    entries: List[Dict[str, Any]] = []
    buffer: List[str] = []
    for line in lines + [""]:  # trailing empty line flushes the last block
        if line.strip():
            buffer.append(line)
            continue
        if buffer:
            try:
                entry = _parse_block(buffer, filepath)
                if entry:
                    entries.append(entry)
            except (ValueError, TypeError) as error:
                logger.warning(f"Skipping block in {filepath.name}: {error}")
            buffer = []

    if not entries:
        logger.warning(f"No valid subtitle entries found in {filepath.name}.")
    return entries


def convert_srts_to_json(root_dir: Path) -> None:
    """Writes a .json next to every .srt below root_dir, skipping the ones already converted."""
    created = existing = failed = 0

    for srt_path in root_dir.rglob("*.srt"):
        json_path = srt_path.with_suffix(".json")
        if json_path.exists():
            existing += 1
            continue

        entries = parse_srt_file(srt_path)
        if not entries:
            logger.warning(f"No entries parsed from {srt_path.name}, no JSON written.")
            failed += 1
            continue

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False, indent=2)
        created += 1

    logger.info(
        f"SRT to JSON: {created} created, {existing} already present, {failed} failed."
    )


def load_subtitles_text(json_path: Path) -> str:
    """Concatenates the dialogue of an episode into a single block of text."""
    with open(json_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        raise ValueError(f"Invalid subtitle JSON (not a list): {json_path}")

    return "\n".join(
        str(entry["text"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("text")
    )


def demo() -> None:
    sample = (
        "1\n00:00:01,000 --> 00:00:03,500\n<i>Hello</i>\nthere\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\nSecond line\n"
    )
    tmp = Path("_demo.srt")
    tmp.write_text(sample, encoding="utf-8")
    try:
        entries = parse_srt_file(tmp)
        assert len(entries) == 2, entries
        assert entries[0]["text"] == "Hello there", entries[0]
        assert entries[0]["start"] == 1.0 and entries[0]["end"] == 3.5
        assert load_subtitles_text.__doc__
        print("srt_parser demo OK")
    finally:
        tmp.unlink()


if __name__ == "__main__":
    demo()
