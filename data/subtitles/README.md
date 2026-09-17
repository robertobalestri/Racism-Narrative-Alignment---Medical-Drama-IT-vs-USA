# Subtitle corpus

The subtitle files are not distributed with this repository for copyright reasons.
To re-run the pipeline, place them here using the following layout, one directory
per dataset:

```
data/subtitles/IT/<Series Name>/S01/E01/<Series Name>S01E01.srt
data/subtitles/US/<SERIES ABBREVIATION>/S01/E01/<ABBREVIATION>S01E01.srt
```

Rules that the pipeline relies on:

- The season directory is `S` followed by two digits, the episode directory is `E`
  followed by two digits.
- The file name is the series name (or abbreviation) immediately followed by
  `S<season>E<episode>`, with no separator. This string is the `episode_code` used
  throughout the results.
- Only `.srt` files are needed. The pipeline writes a `.json` next to each `.srt`
  on its first run and reuses it afterwards.

The corpus behind the published results consists of 1,567 episodes of US medical
dramas and 249 episodes of Italian medical dramas.
