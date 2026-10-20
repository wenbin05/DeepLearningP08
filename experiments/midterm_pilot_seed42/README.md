# Supplied single-seed midterm pilot

The project team uploaded `completed_pilot_outputs.json` and
`preliminary_table.md` from its Colab execution. These files are retained unchanged;
their internal Drive paths identify the original exports. They contain four
completed **validation** records, not full test results.

Protocol: BART-base at the recorded immutable Hub revision; seed 42; fixed data
seed 42; 1,024 training examples; 32 updates; warmup 4; FP16 on one Tesla T4;
64 CNN/DailyMail validation examples; official stemmed Evaluate ROUGE macro F1 ×100
against full original references. All isolated placements target Q/V at rank 8,
with 147,456 trainable parameters each. Zero-shot performs no optimization.

The configured `seeds` list also contains 123 and 999, but **only manifest seed 42
has an actual record**. That configuration list is not evidence of extra runs.

## Checks executed

- Four distinct modes, completed status, validation phase, expected seed/settings.
- Identical generation/model/metric configuration, environment, and data metadata.
- Identical training-example hashes among adapted models and identical validation
  hashes across all four records.
- Independently reloaded the pinned official splits and reproduced both subset-ID
  hashes; recorded in `dataset_hash_verification.json`.
- All three actual target lists match the expected 12 region-specific projections;
  exact trainable/total counts match BART-base plus the specified adapters.
- Configured 32 updates and one epoch agree with saved Trainer timing/step statistics.
- Rounded Markdown scores, losses, and wall times match JSON exactly.
- The interpreter/package/hardware fields agree: Python 3.12.12, PyTorch 2.5.1,
  Transformers 4.44.2, PEFT 0.13.2, Datasets 3.1.0, Evaluate 0.4.3, CUDA 12.4, T4.

These are consistency/provenance checks. Predictions, checkpoints, per-example
scores, and `trainer_state.json` were not uploaded, so this audit did not rerun GPU
training, recompute ROUGE, inspect generated summaries, or independently replay
optimizer updates. No test/XSum or factuality outputs were supplied. The configured
seed list is not treated as independent runs.

## Reproduce report artifacts

From the repository root with the project dependencies installed:

```bash
python -m scripts.prepare_midterm_report
python -m src.plot_results --summary experiments/midterm_pilot_seed42/summary.json \
  --phase validation --output experiments/midterm_pilot_seed42/rouge_vs_parameters.png
pdflatex -interaction=nonstopmode -halt-on-error midterm-template.tex
pdflatex -interaction=nonstopmode -halt-on-error midterm-template.tex
```

The preparation script checks the exported records, extracts `runs/`, uses the
existing aggregation function with expected seed 42 only, and regenerates
`results_table.tex`, `summary.json`, and `verification.json`. Standard deviations
are `null`, not zero. The plot identifies single-seed scores and does not claim
uncertainty estimates. Its score axis is zoomed; differences are very small.
The report uses the generated table directly.

## Interpretation boundaries

Decoder self-attention has the highest observed ROUGE-1/2/L in this small sample;
cross-attention has the lowest loss. Their ROUGE-1 gap is 0.0094 points. All three
adapted models reduce validation loss, but ROUGE changes relative to zero-shot
are small and mixed. No statistical significance, convergence, robust placement
ranking, baseline reproduction, or factual consistency conclusion is supported.

Recorded training-loop seconds include epoch validation and checkpoint saving,
exclude initial preparation, and exclude the separately timed final validation
prediction pass. They are not isolated optimization throughput measurements.
Full/Prefix/all-attention baselines, two more seeds, longer training, rank/projection
ablations, full test/transfer evaluation and the human study remain outstanding.

## Report validation executed

Artifact preparation and plotting commands completed successfully. The existing
result/plotting test module passed **3 tests** (dependency deprecation warnings).
Python compilation and `git diff --check` completed. The updated report compiled
twice with resolved citations/references; the eight-page PDF's methodology and
result-table pages were visually inspected. Minor header-width/caption warnings
remain, with no compilation error. No new GPU training or ROUGE replay was run.
