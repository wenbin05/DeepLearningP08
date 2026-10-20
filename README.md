# CISC 473 P08: Abstractive Summarization via Parameter-Efficient Fine-Tuning

Choo Wen Bin and Nai Junn Rong · [Repository](https://github.com/wenbin05/DeepLearningP08)

Research question: does LoRA placement in BART-base's encoder self-attention,
decoder self-attention, or encoder–decoder cross-attention affect summarization
quality when trainable capacity is held fixed?

At rank 8 on Q/V, each isolated region has **147,456 trainable parameters**.
All-attention Q/V LoRA has 442,368. Only the three isolated placements have matched
parameter budgets. Full fine-tuning has 139,420,416; zero-shot has zero trainable
parameters. Prefix Tuning with 20 virtual tokens has 184,320. The counting command
verifies actual instantiated PEFT parameter shapes using meta tensors, without
pretrained downloads. Saved manifests count real loaded models, and report percentages
against total parameters **including** adapters (147,456 / 139,567,872 ≈ 0.10565%).

**Evidence status:** the team supplied four completed single-seed Colab pilot records:
zero-shot plus the three isolated placements. They use 1,024 training examples,
32 updates, and the same 64 validation examples. The revised report and
[retained artifacts](experiments/midterm_pilot_seed42/README.md) describe these
measurements and limitations. Full-baseline reproduction, multi-seed uncertainty,
full test/transfer results and factuality evaluation remain outstanding. Earlier
unsupported benchmark scores and examples remain excluded. See
[the audit](docs/AUDIT.md) for the original discrepancies. The proposal remains a
historical plan, not a validated experiment record.

## Setup and checks

Python 3.10–3.12; run every command from the repository root. A CUDA GPU is strongly
recommended for the research matrix. CPU checks use a small vocabulary and short
sequences; full data/model runs require substantially more time, RAM and storage.

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only machines: install the smaller wheel first. Omit this line for CUDA.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip check
python -m src.count_params --live
python -m pytest -q
```

Use `python -m src.train`, not `python src/train.py`, so package imports resolve.
The first metric check downloads the pinned official Hugging Face Evaluate ROUGE
module. No alternate or approximate metric is substituted if loading fails.
Tests require that small download once (subsequent runs can use the cache). Model
and dataset downloads are required for experiments, not the synthetic CPU smoke tests.
The dependency versions used for each run are saved; deterministic operations are
requested, but exact results are not guaranteed across devices or library versions.

## Run on Google Colab

Open [train_colab.ipynb in Colab](https://colab.research.google.com/github/wenbin05/DeepLearningP08/blob/main/train_colab.ipynb).
Select **Runtime → Change runtime type → T4 GPU**, then run the cells in order.
The launcher mounts Google Drive, installs its own Python 3.12.12 interpreter and
an isolated environment even when Colab uses a newer Python kernel, pins the tested
code/model/data revisions, and saves a complete dependency lock for later sessions.
Results and logs persist in `MyDrive/CISC473_P08/midterm_fp16_v1`.

Run the smoke cell first, then select one configuration and seed in the research
cell. Repeat seeds 42/123/999 for all seven configurations, or deliberately select
`all_remaining` to run them sequentially. Completed runs and evaluations are skipped;
interrupted training is preserved and requires the explicit retry steps in the
notebook. Automatic mid-epoch resume is not implemented. Use the same GPU type for
all comparisons, and keep smoke outputs outside the research results directory.
The complete matrix can exceed a free Colab session; do not rely on `/content`
for persistent experiment artifacts. The notebook must be run in the user's Colab
session: its GPU and Drive authorization cannot be controlled from this chat.

## Deadline-day preliminary pilot

`midterm_pilot_colab.ipynb` provides a bounded alternative to the full matrix:
zero-shot plus the three isolated LoRA placements, seed 42, a fixed 1,024-example
training subset, 32 updates with warmup 4, and the same 64 validation examples.
An optional cell evaluates 64-example test subsets in each domain without training.
All-attention LoRA, full FT and Prefix Tuning can be added if time permits.

If setup already finished in the main notebook, `scripts/run_midterm_pilot_colab.py`
can be executed in that notebook's namespace instead of repeating installation.
It uses the existing GPU/Drive connection and writes a separate pilot directory,
then exports `preliminary_table.md` and `completed_pilot_outputs.json` from actual
completed runs. It does not overwrite the full research outputs. Automatic retry
of interrupted training remains unsupported; incomplete runs are preserved.

Report these as **single-seed, short-training, validation-subset pilot results**.
There is no standard deviation, convergence finding, full-test benchmark, or
published-baseline reproduction claim. The remaining seeds, guide-required
baselines/ablations and human study are still outstanding. This reduced execution
plan does not establish that the instructor waived any rubric requirement.
GPU duration has not been measured here; use the first run's progress to plan time.

## Reproduce experiments

Complete baseline/placement matrix: 7 configurations × 3 seeds, then both test domains,
aggregation, and plots:

```bash
bash scripts/run_midterm.sh
```

The script resolves immutable model and dataset revisions once for the entire matrix.
It trains on a fixed 10,000-example CNN/DailyMail subset (data seed 42), validates
on the full official validation split, and evaluates each final checkpoint on full
CNN/DailyMail and XSum test splits. Seeds **42, 123, 999** independently initialize
adapters and training randomness; data membership stays fixed. Full fine-tuning
uses learning rate 3e-5; adapters use 3e-4. All trained configurations use the same
examples, 3 epochs, and effective batch size 32 on a single device. Equal updates
are an exposure control, **not evidence of matched FLOPs or elapsed compute**.
Compare saved training time/update counts when reporting the guide's compute-budget
requirement; a strict wall-clock/FLOP comparison remains a separate experiment.

For the guide's rank and projection ablations too:

```bash
EXTENDED_ABLATIONS=1 OUTPUT_DIR=results/extended bash scripts/run_midterm.sh
```

This adds all-attention Q/V ranks 4 and 16 (rank 8 already exists), and rank-8
Q-only, K-only, V-only, and Q/K/V/O runs. These comparisons deliberately vary capacity;
they are separate from the matched-budget placement question. This is a large matrix,
not a quick local test. Do not claim it ran merely because the script exists.

Independent pilot, then held-out evaluation (no further fine-tuning):

```bash
python -m src.train --placement cross_attn --seed 42 --output_dir results/pilot
python -m src.train --eval_only results/pilot/cross_attn_r8_q_proj,v_proj_seed42 \
  --eval_dataset cnn_dailymail
python -m src.train --eval_only results/pilot/cross_attn_r8_q_proj,v_proj_seed42 \
  --eval_dataset xsum
```

Each invocation refuses to overwrite its run/evaluation directory. Use distinct
`--output_dir` roots for protocols with different sample limits, prefix lengths,
epochs, or precision. For a short **execution check**, use e.g.
`--max_train_samples 8 --max_eval_samples 2 --max_steps 1 --warmup_steps 0`.
This still downloads BART-base and the dataset. Such outputs are pilot results,
not full-test benchmarks. During training `--max_eval_samples` limits validation;
with `--eval_only` it limits test. Omit it for a full official evaluation. Use `--max_train_samples all` to train on
the full official training split; report that as a separate protocol from the 10k pilot.
`--fp16` explicitly enables mixed precision on CUDA; default is FP32.

Additional controls: `--lora_rank`, `--projections` (comma-separated PEFT projection
names), `--prefix_tokens`, `--num_train_epochs`, `--max_steps`, `--warmup_steps`,
`--data_seed`, `--model_revision`, `--dataset_revision`; test data revision uses
`--eval_revision`. `src/config.py` contains shared batch sizes and length limits.
Warmup 500 is the research default; reduce it explicitly for short pilots.

## Saved artifacts and evaluation protocol

Each run saves:

- `run_config.json`: experiment/model/Trainer/generation configuration, exact target
  modules, parameter counts, pinned model/data/metric revisions, selected-example
  hashes, seeds, package versions, hardware, and Git revision/dirty status.
- `model/`: final full model or PEFT adapter plus tokenizer. Keep `run_config.json`
  with adapters: evaluation loads the original base checkpoint at its pinned SHA.
- `validation_results.json`: ROUGE-1/2/L, validation loss, training metrics and wall
  time in seconds. Epoch checkpoints and `trainer_state.json` retain training history.
- `validation_predictions.jsonl`: IDs, source text, generated text, original and
  truncated references. Test evaluation saves the same fields under
  `evaluations/<dataset>_<example-hash>/predictions.jsonl` and `test_results.json`.

CNN/DailyMail training uses official `train`, validation uses `validation`; final
in-domain evaluation and XSum transfer use official `test`. Inputs truncate to
1024 BPE tokens; training labels to 128, with dynamic padding and `-100` loss masks.
ROUGE uses **full original references**, stemming, and macro mean per-example F1
×100 from Hugging Face Evaluate. It reports `rougeL`, not `rougeLsum`; do not assume
comparability with papers using a different metric implementation or protocol.

Generation uses beam width 4, length penalty 2, min length 56, max length 128,
3-gram blocking, and no sampling. The same settings apply to both test domains.
Minimum length 56 can be a poor fit for XSum's single-sentence summaries; changing
it must be labeled as a distinct evaluation ablation. Neither domain is used to
select training hyperparameters or checkpoints. Current selection is final epoch/update.

Aggregate genuine completed files and plot them:

```bash
python -m src.aggregate_results --results_dir results/pilot --phase test \
  --output results/pilot/summary.json
python -m src.plot_results --summary results/pilot/summary.json \
  --output results/pilot/cnn_pareto.png
python -m src.plot_results --summary results/pilot/summary.json \
  --dataset xsum --output results/pilot/xsum_pareto.png
```

Aggregation requires exactly one run per expected seed and compatible within-group
protocols; it rejects missing seeds, duplicates and mixed subsets/settings. Standard
deviation uses `ddof=1`. For explicitly labeled single-seed pilots, pass `--seeds 42`;
standard deviation is then unavailable (`null`), not zero. Plots use saved values
and include zero-shot on a symlog parameter axis. No expected placement ranking is encoded.

## Report and remaining work

`midterm-template.tex` and its rebuilt PDF now report the supplied single-seed pilot.
The decoder--cross ROUGE-1 gap is about 0.01 points on 64 validation examples;
no reliable ranking or multi-seed uncertainty is inferred. Validation loss improved
for all three placements while generated-summary ROUGE changes were mixed.
Full fine-tuning, Prefix Tuning, all-attention comparisons, additional seeds,
rank/projection ablations, full CNN/DailyMail/XSum tests, and the human study still
require genuine research outputs. Only the four validation records were supplied;
no optional test or transfer results are claimed. Predictions and checkpoints were
not supplied, so ROUGE was checked against the export/table but not recomputed.

Regenerate the pilot's checked table, per-run artifacts, single-seed summary and plot:

```bash
python -m scripts.prepare_midterm_report
python -m src.plot_results --summary experiments/midterm_pilot_seed42/summary.json \
  --phase validation --output experiments/midterm_pilot_seed42/rouge_vs_parameters.png
```

The human study requires 10 participants and 20 summaries. Prediction export is
implemented, but participant collection, blinding and analysis remain future work.
The report's AI disclosure covers code, Colab troubleshooting, artifact consistency
checks and drafting; the team executed the GPU pilot and supplied its results.

Rebuild the report (LaTeX distribution with the packages in the source):

```bash
pdflatex -interaction=nonstopmode -halt-on-error midterm-template.tex
pdflatex -interaction=nonstopmode -halt-on-error midterm-template.tex
```

The report includes an AI Usage disclosure for this audit/revision; the team must
review it and add any earlier AI use. The notebook uses the same repository commands
and requirements rather than a parallel implementation. Commit the review branch
before research runs so manifests record a clean reproducible Git revision.
