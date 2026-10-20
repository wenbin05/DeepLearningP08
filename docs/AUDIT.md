# P08 implementation and report audit

Update after the initial audit: the team has now supplied four completed seed-42
Colab pilot records, retained under `experiments/midterm_pilot_seed42`. The revised
report includes those measured validation-subset results. The sections below record
the original audit and its initial evidence state; they do not invalidate the new
pilot. Full benchmark reproduction, extra seeds, required baseline/ablation runs,
test/transfer and human-study evidence remain outstanding. See that directory's
README for the subsequent artifact checks and their limits.

Audit input: repository at `cc03f50`, all six original Python scripts,
`train_colab.ipynb`, requirements, README, proposal, and midterm LaTeX/PDF; attached
*CISC 473 Capstone Project Guide, Fall 2026*, reproducibility requirements,
midterm rubric, and P08 catalogue entry (printed page 14). The guide is a rubric
reference, not an instruction overriding the user's request. No instructor feedback,
training checkpoints, raw results, or logs were supplied. The repository's earlier
claims cannot be authenticated from their repetition in a PDF or README.

## Confirmed issues and dispositions

| Finding in original implementation | Fix / evidence boundary |
|---|---|
| README/notebook invoke scripts directly; `from src...` then fails without repo root on `sys.path` | Document and use `python -m src...`; shared package marker added |
| Trainer never has `compute_metrics`; imported ROUGE unused | Official metric callback connected for validation and test |
| Broad `except Exception` silently fabricates fallback metrics; ROUGE-L is scaled unigram score; repeated n-grams can exceed valid counts | Entire fallback removed; official pinned Evaluate metric; errors propagate |
| Report and plot scores have no per-seed artifacts; figure hard-codes scores and conclusions | Figure removed; plotting reads aggregated saved results; report quality claims withheld |
| Final metrics merely printed; no controlled model export or configuration provenance | Atomic completed-result JSON, full manifest, Trainer state, final checkpoint/tokenizer, prediction JSONL |
| Only first 500 validation rows evaluated; report says full test | Official split selection, deterministic shared subsets, hashes/sizes, explicit validation/test result names |
| Generation min length, penalty, and n-gram block unused | Shared GenerationConfig, also assigned to PEFT's delegated base model |
| Duplicate model loading/injection logic | Shared loader and actual module/count assertions |
| Hard-coded analytical assertion does not exercise PEFT | Instantiate BART-base module tree with PEFT on meta tensors; also real forward/backward/save/reload smoke tests |
| `fp16` config ignored; zero-shot model marked trainable | Explicit CUDA-only opt-in; freeze zero-shot parameters |
| No XSum/test checkpoint evaluation | Evaluation-only path restores pinned base and adapters/full models; no optimizer/training |
| Prefix Tuning and rank/projection CLI absent | Shared Prefix Tuning loader, rank/projection controls, optional ablation matrix |
| Unbounded dependencies omit plotting/testing libraries | Pinned compatible stack including transitive Arrow/Hub constraints, CI CPU checks |
| No tests, notwithstanding report claims | Tests added; actual local outcomes below |

## Rubric coverage and scope

| Requirement | Original state | Current implementation / execution status |
|---|---|---|
| Working baseline end to end (implementation, 8 points) | Training scaffold; no ROUGE results | Training/evaluation/save/reload covered by CPU smoke tests; real baseline training still required |
| Baseline reproduction + preliminary ablations (results, 8 points) | Unsupported numerical table | Baselines + three placements executable; no genuine benchmark scores yet |
| Comparative literature (6 points) | Overstated He/DyLoRA interpretation; incorrect DyLoRA metadata | Narrowed claims, corrected citation; authors must maintain critical review and verify comparison protocols |
| Obstacles/updated plan (4 points) | Unsupported OOM/bias/test observations | Replaced with confirmed audit defects and outstanding execution plan |
| Writing/presentation (4 points) | Results presented as established | Academic structure preserved; evidence status made explicit; PDF rebuilt |
| ≥3 seeds, single command, dependencies | Only seed 42 examples; no aggregation | Matrix 42/123/999; mean/sample SD from saved files; script and pinned requirements |
| LoRA rank 4/8/16, Q/K/V/all ablation | Only rank-8 Q/V targets | CLI and optional extended matrix; experiments outstanding |
| Prefix/full FT under matched compute | Prefix absent; full FT scaffold | Both executable; same examples/updates, timing saved; strict equal-time/FLOP comparison remains outstanding |
| XSum with no further tuning | Not implemented | Saved-model test path and matrix evaluation; genuine transfer runs outstanding |
| Human factual consistency, 10 participants × 20 summaries | No evidence | Prediction/source export supports preparation; collection/analysis outstanding |

The draft asserts a faculty-approved narrowing of the proposal. No supporting
feedback was supplied, so it cannot justify declaring original P08 requirements
waived. The user explicitly prioritizes placement; code preserves that question
while exposing the other guide-required experiments. The guide calls for groups
of three, while this repository lists two members; administrative approval for
that difference also cannot be verified here. Do not invent a third contributor.

## Report claims requiring evidence

Removed/withheld from the current report:

- All ROUGE means/SDs for zero-shot, full FT, all-attention, encoder-only,
  decoder-only and cross-attention configurations, including 42.54 ± 0.18 and
  41.42 ± 0.16 ROUGE-1; the 98.9% retention and +1.60-point placement advantage.
- Published “BART-base 42.30/19.50/39.40” attribution and “within 0.5%” reproduction.
  A matching primary source/model/split/metric protocol was not established.
  Also distinguish ROUGE points from relative percentage errors.
- FAA article/reference/generated examples: no article ID or checkpoint provenance.
- CUDA OOM events, performance tuning, prior unit-test runs, and multi-seed execution.
- Conclusions that cross-attention is superior or that encoder adaptation is unnecessary.
- Claimed approval of reduced scope; actual research contribution percentages and
  historical AI use require the team's confirmation.

Technical corrections: BART-base has **12**, not 16 attention heads; PEFT LoRA A
uses Kaiming-uniform initialization by default, with zero B; XSum's 204,045 count
is its training split, not total (total 226,711). Base-denominator percentages differ
slightly from total-with-adapter percentages; saved manifests state the latter.
Default data use is a fixed 10k pilot subset, not all 287,113 training examples.
Equal trainable budgets apply only to the three isolated placements.

Primary references checked for citation audit:

- [BART-base configuration](https://huggingface.co/facebook/bart-base/blob/aadd2ab0ae0c8268c7c9693540e9904811f36177/config.json)
- [DyLoRA](https://aclanthology.org/2023.eacl-main.239/): corrected title and pages 3274–3287;
  removed unsupported encoder/decoder specialization interpretation.
- [WeightLoRA](https://aclanthology.org/2026.acl-long.566/): title/authors/venue/pages exist;
  its existence is not evidence for this project's numerical conclusions.
- [FactCC](https://aclanthology.org/2020.emnlp-main.750/): domain-sensitive factuality
  evaluation motivates future human evaluation; no FactCC run has been performed here.

The historical proposal and its compiled PDF are retained unchanged. Its detailed
literature numbers should be checked against primary tables and evaluation protocols
before reuse; this audit does not certify them as experiment outputs. The revised
midterm avoids importing those numbers as local findings. AI disclosure explicitly
covers this revision and does not guess the authors' earlier AI usage.

## Execution and resource limits

This environment has CPU PyTorch and no CUDA GPU. Research-scale fine-tuning,
full held-out CNN/DailyMail/XSum evaluation, three-seed comparisons, and human data
collection were not performed. Run the matrix on an external GPU or suitably
provisioned CPU host; the guide's indicative timing is not a measurement here.
No throughput, peak memory, FLOP, or benchmark-score claims are inferred from tests.
Download/cache space for datasets and full checkpoints is required. Free Colab
sessions may be too short for the complete matrix; persist each run directory.

## Validation executed on this branch

- `python -m pytest -q`: **29 passed**, 46 dependency warnings, 19.24 seconds in
  the final run on Python 3.12.14 / PyTorch 2.5.1+cpu. Real official ROUGE was used.
- Smoke coverage: one optimizer update, finite loss/ROUGE, parameter changes,
  generation, save/reload output equality for the three LoRA placements and Prefix
  Tuning, full fine-tuning, and zero-shot inference. LoRA smoke models retain the
  six-layer/768-dimensional BART architecture with a tiny synthetic vocabulary;
  other modes use tiny random BART models. These are software tests, not benchmarks.
- Production CLI orchestration test: saves manifests, results, state, and a model;
  evaluates both test schemas; explicitly fails if evaluation calls `Trainer.train`.
- Live PEFT shape counts: each isolated region **147,456**, all-attention **442,368**,
  20-token Prefix Tuning **184,320**, full FT **139,420,416**, zero-shot **0**.
- `pip check`: no broken requirements. The initial install exposed a NumPy/PyArrow
  incompatibility; compatible versions are now pinned. Initial smoke tests exposed
  a Transformers 4.46.3 / PEFT 0.13.2 Prefix generation incompatibility; the tested
  stack uses Transformers 4.44.2 rather than a custom monkey patch.
- Actual Hub streaming probes read one official test example each from
  CNN/DailyMail SHA `96df5e686bee6baa90b8bee7c28b81fa3fa6223d` and XSum SHA
  `7d4d486c2f8ef850b1a11aead99b894ff3dd7da9`, confirming expected source/reference
  columns. No dataset-wide metric was calculated by these probes.
- `python -m compileall -q src tests`, CLI help, `bash -n scripts/run_midterm.sh`,
  and `git diff --check` completed successfully.
- `pdflatex -interaction=nonstopmode -halt-on-error` ran twice; rebuilt seven-page
  PDF has resolved citations/references. First page visually inspected. Minor header
  box/caption-package warnings remain; compilation succeeded.

CI is configured to rerun CPU checks after push; no CI outcome is claimed here.
No GPU, research training matrix, full benchmark evaluation, or human study was run.
