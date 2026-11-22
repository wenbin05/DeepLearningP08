# Human factual-consistency evaluation

This is a protocol and collection tool, not completed participant evidence.
It operationalizes the guide's 10 participants / 20 summaries as five randomly
selected held-out articles × four configurations: zero-shot and three isolated
placements, all training seed 42. Each of ten participants rates all twenty
summaries. This design compares the core placements but cannot characterize
seed variability or the omitted baselines. Five articles also limit generalization.

First complete held-out prediction export and verification. Use the four
CNN/DailyMail `predictions.jsonl` paths with the same evaluation subset/protocol:

```bash
python -m scripts.human_evaluation prepare --predictions \
  <zero-shot-predictions> <encoder-predictions> <decoder-predictions> <cross-predictions> \
  --output_dir results/human_study
```

Replace the angle-bracket placeholders with actual paths. Preparation verifies
recorded scores against predictions, shared protocols and example IDs before
sampling with a fixed random seed. It refuses to overwrite an existing packet.

Give each participant only their `participants/Pxx` folder, containing readable
`packet.html` and blank `ratings.csv`. Keep `coordinator/key.json` private until
ratings are complete. Each packet uses a different randomized order and contains
source articles, generated summaries, anonymous summary IDs and instructions.
Do not select articles based on favorable results or identify models to raters.
Participants should rate independently without discussing answers.

Use three judgments: consistent, inconsistent, uncertain. Consistent means all
factual claims are supported by the article; inconsistent means at least one is
unsupported or contradicted. Missing coverage or poor writing alone is not factual
inconsistency. Require a source-based explanation for an inconsistent judgment.
Keep uncertain distinct from a positive judgment. Use anonymous participant IDs,
voluntary participation, and applicable course guidance; do not collect names in
the ratings files. The project team must recruit participants and collect genuine
judgments. No contacts are messaged by these tools.

After collecting the ten completed files:

```bash
python -m scripts.human_evaluation analyze \
  --key results/human_study/coordinator/key.json \
  --ratings results/human_study/participants/P*/ratings.csv \
  --output results/human_study/analysis.json
```

Analysis rejects missing, duplicate or invalid ratings. It reports category counts,
the consistency fraction among decided ratings, pairwise agreement and three-category
Fleiss' kappa (undefined when expected agreement is one). Inspect disagreements and
retain representative source-grounded errors. Report denominators and uncertain
ratings, not only a single percentage. Treat all statistics as descriptive: shared
articles and raters create dependence, and 200 ratings are not 200 independent
examples. This does not demonstrate a statistically reliable model ranking.
