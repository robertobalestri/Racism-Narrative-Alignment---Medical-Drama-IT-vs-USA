# Racism Narrative Alignment

Software used to measure how television episodes position themselves with respect
to racism, and to produce the results and figures reported in the accompanying
study. Every LLM call in the pipeline is made to GPT-4.1.

Two corpora are analysed with the same method: US medical dramas (1,553 episodes)
and Italian medical dramas (249 episodes), all first broadcast between 1994 and
2024.

## Method in three stages

**1. Screening.** Each episode's subtitles are sent to GPT-4.1 with a short prompt
asking whether racism, racial discrimination or racial prejudice is mentioned or
implied, explicitly or otherwise. The instruction is deliberately permissive: in
case of doubt, answer Yes. 416 US episodes and 66 Italian episodes pass this
filter.

**2. Likert evaluation.** Every episode that passed screening is scored on a scale
from -3 (active endorsement of discrimination) to +3 (transformative critique of
systemic racism), with "N/A" available when the topic turns out to be absent after
all. The model returns the score, an explanation, and a justification for not
picking the adjacent lower and higher score. **Each episode is evaluated ten
times**, so that the stability of the judgement can be measured rather than
assumed.

**3. Air dates.** The original broadcast date of each episode, curated by hand, is
merged into the results. It is the time axis of the diachronic analysis.

The two corpora use different evaluation prompts. The Italian prompt is written in
Italian and frames racial issues around migrants, Roma communities and
second-generation Italians, which is what the Italian material actually depicts;
the US prompt uses the original English scale. Both share the same scale range and
the same instructions about avoiding a default score of 0.

## Layout

```
prompts/                 The three prompts, verbatim as used for the published runs
src/                     Configuration, GPT-4.1 client, SRT parsing
pipeline/                The three stages, in order
analysis/                Stability statistics and the five figure scripts
data/airdates/           Curated broadcast dates, one row per episode
data/subtitles/          Where the corpus goes (not distributed; see its README)
results/screening/       Stage 1 output: which episodes are relevant
results/likert/          Stage 2 output: ten evaluations per episode
results/final/           Stage 3 output: evaluations plus air dates
results/figures/         The figures, each with a same-named .md: numbers and method
```

The folder is self-contained: it imports nothing from outside itself.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your Azure OpenAI credentials
```

Stages 1 and 2 need the subtitle corpus in `data/subtitles/` (see the README
there) and make paid API calls. Stage 3 and everything under `analysis/` run on
the results shipped in this repository, with no API access and no corpus:

```bash
python pipeline/01_screen_episodes.py   --dataset IT     # or US
python pipeline/02_likert_evaluation.py --dataset IT
python pipeline/03_merge_airdates.py    --dataset IT

cd analysis
python stability_stats.py
python plot_evolution.py
python plot_gam.py
python plot_distribution.py
python plot_lustrums.py
```

Stages 1 and 2 save after every episode and skip work already present in their
output CSV, so an interrupted run can simply be restarted.

## Reading the results

`results/final/dati_racism_{IT,US}_con_data.csv` is the file the analysis reads.
It is semicolon-separated and UTF-8 with BOM, so that it opens directly in an
Italian Excel locale. One row is one evaluation, so each episode occupies ten
consecutive rows.

Two points matter when interpreting the numbers:

- **"N/A" is not zero.** An evaluation of N/A means the model found no racial
  content in the episode after reading it in full, despite the permissive
  screening. They account for 22% of the Italian evaluations and 47% of the US
  ones.
- **The final sample is the valid episodes.** An episode is kept only if at most
  5 of its 10 evaluations are N/A (`MAX_NA_EVALUATIONS` in `src/config.py`):
  50 of 66 Italian episodes and 222 of 416 US ones. Every figure, mean, trend and
  distribution uses these episodes only, and within them only the numeric
  evaluations.
- **Ten evaluations, not one.** Any statistic computed over the rows of this file
  weights each episode ten times. `analysis/stability_stats.py` reports how much
  those ten answers actually differ, and writes a per-episode summary to
  `results/final/stability_{IT,US}.csv`; `analysis/stability_table.py` writes the
  same indicators as the Italian table below to `results/final/stability_table.md`.
  `analysis/series_dispersion.py` writes the spread of episode medians within each
  series to `results/final/series_dispersion.md`.

Stability of the published runs, at temperature 0.3:

| | Italian corpus | US corpus |
|---|---|---|
| Episodes that passed screening | 66 | 416 |
| All ten evaluations identical (ten N/A count as identical) | 59.1% | 67.1% |
| Episodes mixing N/A with a score | 19.7% | 23.6% |
| Valid episodes (at most 5 N/A) | 50 | 222 |
| Valid episodes with an interquartile range of 0 | 84.0% | 86.9% |
| Mean interquartile range of numeric scores, valid episodes | 0.195 | 0.179 |
| Mean range (max - min), valid episodes | 0.360 | 0.378 |
| Mean standard deviation of numeric scores, valid episodes | 0.157 | 0.167 |

## Reproducibility notes

- Temperature is 0.3. Raising it widens the spread across the ten evaluations;
  lowering it narrows it. The published figures assume 0.3.
- `prompts/racism_likert_prompt_US.txt` is reproduced exactly as it was used,
  including a few lines of conversational preamble at the top of the file that
  were left there while the prompt was being drafted. They are part of what the
  model received and are kept for fidelity.
- Air dates are input data, not something the software derives. Stage 3 merges
  `data/airdates/airdates_*.csv` onto the Likert results by episode code.
  It warns if any merged air date falls outside the 1994-2024 study period.
- Running stage 3 on the shipped Likert results reproduces the shipped final CSVs
  exactly; running the analysis scripts reproduces the shipped figures.
