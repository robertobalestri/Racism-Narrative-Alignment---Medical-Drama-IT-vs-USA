"""Paths, dataset definitions and model settings for the whole pipeline."""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"
SUBTITLES_DIR = DATA_DIR / "subtitles"
AIRDATES_DIR = DATA_DIR / "airdates"
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENING_DIR = RESULTS_DIR / "screening"
LIKERT_DIR = RESULTS_DIR / "likert"
FINAL_DIR = RESULTS_DIR / "final"
FIGURES_DIR = RESULTS_DIR / "figures"

# Number of independent LLM evaluations per episode, used to measure stability.
NUMBER_OF_ATTEMPTS = 10

# An episode stays in the final sample only if at most this many of its ten
# evaluations are "N/A".
MAX_NA_EVALUATIONS = 5

# The study covers episodes first broadcast in these years, inclusive.
STUDY_FIRST_YEAR = 1994
STUDY_LAST_YEAR = 2024

GPT41_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("GPT4_1_API_KEY"),
    "api_endpoint": os.getenv("GPT4_1_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "deployment_name": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_GPT4_1"),
    "temperature": os.getenv("AZURE_OPENAI_TEMPERATURE", "0.3"),
}

DATASETS: Dict[str, Dict[str, Path]] = {
    "IT": {
        "subtitles_dir": SUBTITLES_DIR / "IT",
        "likert_prompt": PROMPTS_DIR / "racism_likert_prompt_IT.txt",
        "screening_csv": SCREENING_DIR / "gpt_4_1_racism_classification_results_IT.csv",
        "likert_csv": LIKERT_DIR / "gpt41_likert_episodio_razzismo_full_IT.csv",
        "airdates_csv": AIRDATES_DIR / "airdates_IT.csv",
        "final_csv": FINAL_DIR / "dati_racism_IT_con_data.csv",
    },
    "US": {
        "subtitles_dir": SUBTITLES_DIR / "US",
        "likert_prompt": PROMPTS_DIR / "racism_likert_prompt_US.txt",
        "screening_csv": SCREENING_DIR / "gpt_4_1_racism_classification_results_US.csv",
        "likert_csv": LIKERT_DIR / "gpt41_likert_episodio_razzismo_full_US.csv",
        "airdates_csv": AIRDATES_DIR / "airdates_US.csv",
        "final_csv": FINAL_DIR / "dati_racism_US_con_data.csv",
    },
}

SCREENING_PROMPT = PROMPTS_DIR / "racism_screening_prompt.txt"


def dataset(name: str) -> Dict[str, Path]:
    """Returns the path set for a dataset key ('IT' or 'US')."""
    key = name.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[key]

