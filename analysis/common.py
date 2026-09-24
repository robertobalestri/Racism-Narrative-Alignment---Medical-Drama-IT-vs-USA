"""Shared loading helpers for the analysis scripts."""

import sys
from pathlib import Path

import pandas as pd
from matplotlib.ticker import FixedLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, MAX_NA_EVALUATIONS, dataset

COUNTRY_LABELS = {"IT": "Italian Series", "US": "US Series"}

# Display titles, keyed by the series name used in the result files.
SERIES_TITLES = {
    "ER": "ER",
    "H": "House, M.D.",
    "GA": "Grey's Anatomy",
    "PP": "Private Practice",
    "CM": "Chicago Med",
    "TGD": "The Good Doctor",
    "TR": "The Resident",
    "NA": "New Amsterdam",
    "LA DOTTORESSA GIO": "La Dottoressa Giò",
    "MEDICINA GENERALE": "Medicina Generale",
    "TERAPIA D URGENZA": "Terapia d’urgenza",
    "DOC NELLE TUE MANI": "DOC – Nelle tue mani",
    "LEA UN NUOVO GIORNO": "Lea – Un nuovo giorno",
    "BRACCIALETTI ROSSI": "Braccialetti Rossi",
    "CRIMINI BIANCHI": "Crimini Bianchi",
    "CUORI": "Cuori",
    "L ALLIEVA": "L’allieva",
    "LA LINEA VERTICALE": "La Linea Verticale",
    "MENTAL": "Mental",
    "NATI IERI": "Nati ieri",
    "OLTRE LA SOGLIA": "Oltre la soglia",
}

# Original broadcast run of each series, as stated in the paper. Not derived from
# the data: the corpus holds only the episodes that deal with the topic.
SERIES_RUNS = {
    "ER": "1994–2009",
    "House, M.D.": "2004–2012",
    "Grey's Anatomy": "2005–present",
    "Private Practice": "2007–2013",
    "Chicago Med": "2015–present",
    "The Good Doctor": "2017–2024",
    "The Resident": "2018–2023",
    "New Amsterdam": "2018–2023",
    "La Dottoressa Giò": "1997–1998, 2019",
    "Medicina Generale": "2007–2010",
    "Terapia d’urgenza": "2008–2009",
    "DOC – Nelle tue mani": "2020–present",
    "Lea – Un nuovo giorno": "2022–2023",
    "Braccialetti Rossi": "2014–2016",
    "Crimini Bianchi": "2008–2009",
    "Cuori": "2021–present",
    "L’allieva": "2016–2020",
    "La Linea Verticale": "2018",
    "Mental": "2020",
    "Nati ieri": "2006–2007",
    "Oltre la soglia": "2019",
}


def load_evaluations(dataset_name: str) -> pd.DataFrame:
    """Loads every evaluation of one dataset, with "N/A" scores as NaN."""
    # keep_default_na=False: pandas would otherwise read the series code "NA"
    # (New Amsterdam) as missing. "N/A" scores are coerced to NaN explicitly.
    frame = pd.read_csv(
        dataset(dataset_name)["final_csv"],
        sep=";",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["series"] = frame["series"].replace(SERIES_TITLES)
    return frame


def load_scores(dataset_name: str) -> pd.DataFrame:
    """Loads the final sample of one dataset: valid, dated episodes, numeric scores only.

    An episode is valid when at most MAX_NA_EVALUATIONS of its ten evaluations are
    "N/A" (the topic is absent); the others are left out entirely. The remaining
    N/A evaluations of valid episodes are dropped as rows.
    """
    frame = load_evaluations(dataset_name)
    na_per_episode = frame["score"].isna().groupby(frame["episode_code"]).transform("sum")
    frame = frame[na_per_episode <= MAX_NA_EVALUATIONS]
    frame["Air Date"] = pd.to_datetime(
        frame["data_messa_in_onda"], dayfirst=True, errors="coerce"
    )
    frame = frame.dropna(subset=["Air Date", "score"]).copy()
    frame["Year"] = frame["Air Date"].dt.year
    frame["Country"] = COUNTRY_LABELS[dataset_name.upper()]
    return frame


def set_year_ticks(axes, years=range(1995, 2026, 5)) -> None:
    """Places x ticks on 1 January of each year, for axes plotted in date ordinals.

    Matplotlib's automatic ticks fall on arbitrary days, so labelling them with the
    year alone would shift the axis by up to a year.
    """
    ticks = [pd.Timestamp(year=year, month=1, day=1).toordinal() for year in years]
    axes.xaxis.set_major_locator(FixedLocator(ticks))
    axes.set_xticklabels([str(year) for year in years])


def decimal_it(value: float, digits: int = 3) -> str:
    """Formats a number with a decimal comma, as in the Italian tables of the study."""
    return f"{value:.{digits}f}".replace(".", ",")


def episode_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per valid episode: its median numeric score and its air date.

    The ten evaluations of an episode are repeated judgements of the same text, not
    independent observations, so statistical tests use one value per episode.
    """
    episodes = (
        frame.groupby("episode_code")
        .agg(
            series=("series", "first"),
            Country=("Country", "first"),
            air_date=("Air Date", "first"),
            evaluations=("score", "size"),
            score=("score", "median"),
        )
        .reset_index()
    )
    day_of_year = episodes["air_date"].dt.dayofyear - 1
    episodes["decimal_year"] = episodes["air_date"].dt.year + day_of_year / 365.25
    return episodes


def corpus_size(dataset_name: str) -> int:
    """Episodes in the corpus, i.e. every episode that went through screening."""
    screening = pd.read_csv(dataset(dataset_name)["screening_csv"], keep_default_na=False)
    return len(screening)


def p_value(value: float) -> str:
    """APA style: no leading zero, and '< .001' for very small values."""
    return "< .001" if value < 0.001 else f"= {value:.3f}".replace("0.", ".", 1)


def write_notes(figure_filename: str, lines: list) -> None:
    """Writes the figure's companion Markdown file next to it, with the same name."""
    output = figure_path(figure_filename).with_suffix(".md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Notes written to {output}")


def figure_path(filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename
