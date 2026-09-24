# Internal stability of the LLMs on the Likert test sample

20 episodes, each evaluated 5 times by every model with the same prompt.
Only the episodes evaluated by all four models are counted.

Per episode and model, the five scores are compared with each other:

- Unanimous: all five scores identical.
- IQR = 0: the middle half of the five scores coincide.
- IQR, range (max - min) and standard deviation (sample, ddof=1) are
  averaged over the episodes. Lower means more stable.

Best value per column in bold. Mean score is shown for context only: it is
not a stability measure.

| Model | Unanimous episodes | Episodes with IQR = 0 | Mean IQR | Mean range | Mean SD | Mean score |
|---|---|---|---|---|---|---|
| gpt-4.1 | **4/20 (20%)** | **13/20 (65%)** | **0.35** | **1.05** | **0.473** | +0.37 |
| gpt-4o-mini | 3/20 (15%) | 9/20 (45%) | 0.55 | 1.35 | 0.606 | +0.57 |
| gemini-2.5 | 3/20 (15%) | 9/20 (45%) | 0.55 | 1.10 | 0.511 | +0.93 |
| deepseek-v3 | 1/20 (5%) | 10/20 (50%) | 0.55 | 1.70 | 0.698 | +0.18 |
