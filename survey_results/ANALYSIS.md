# Exploratory factual-consistency survey

## Collection and design

The project team confirmed that people independently read the source articles and
compared them with the summaries to complete the ratings. This is the team's
reported collection method. The analysis retains all submitted responses unchanged.

Ten participant IDs each have 20 ratings: 200 ratings covering 20 generated
summaries from five shared articles and four configurations. Participants received
different randomized orders and blinded summary IDs. The stimuli came from the
CNN/DailyMail held-out subset, using training seed 42 for the adapted models.
The sampling design does not evaluate factuality across training seeds or on XSum.

These calculations use the latest upload at commit
`fd90ad3c6dc3e109f8dccea845f0b4c44abbb01f`, organized without content changes.
The summary key is now included so configuration-level statistics are reproducible.

## Descriptive results

Each configuration contributes five summaries, each with ten judgments (50 ratings).
No respondent selected `uncertain`.

| Configuration | Consistent ratings | Inconsistent ratings | Uncertain ratings | Consistent / decided ratings |
|---|---:|---:|---:|---:|
| Zero-shot BART-base | 27 | 23 | 0 | 54.0% |
| Cross-attention LoRA | 29 | 21 | 0 | 58.0% |
| Encoder self-attention LoRA | 19 | 31 | 0 | 38.0% |
| Decoder self-attention LoRA | 23 | 27 | 0 | 46.0% |

Across all configurations, 98 of 200 ratings were consistent and 102 were
inconsistent. These percentages describe judgments, not independently verified
accuracy, and are not estimates from 200 independent summaries.

## Agreement

- Mean pairwise agreement within summaries: **52.22%**.
- Fleiss' kappa using the three allowed categories: **0.0441**.
- Each summary has ten judgments. Kappa uses the observed category marginals;
  the unused uncertain category contributes zero to expected agreement.

The agreement is only slightly above chance under this calculation. It may reflect
ambiguity, inconsistent application of the rubric, or differing interpretations.
The files alone do not distinguish these explanations.

## Interpretation and limitations

Cross-attention has the highest observed consistency fraction (58%), followed by
zero-shot (54%), decoder self-attention (46%) and encoder self-attention (38%).
The cross-attention versus zero-shot difference is two ratings out of 50 per
configuration; it is not evidence of a statistically reliable advantage.

Some explanations flag omitted details as factual inconsistencies even though the
instructions distinguish missing coverage from unsupported or contradictory claims.
See REVIEW_NOTES.md for examples. These responses have not been corrected or
removed. Low agreement is retained as a study result, not optimized away.

The small sample of five articles, shared articles and raters, one training seed,
and interpretation issues limit generalization. Do not treat the 200 ratings as
independent observations, claim a validated factuality ranking, or use agreement
as proof that a judgment is correct. No significance test or confidence interval
is claimed here.

## Reproduction

```bash
python survey_results/analyze.py
```

This standard-library script validates the submitted rows and regenerates
[analysis.json](analysis.json). The raw CSVs remain unchanged.
