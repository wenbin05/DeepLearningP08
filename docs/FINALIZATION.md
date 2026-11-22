> **Historical planning document.** The final CPU subset study is now complete.
> See [current status](FINALIZATION_STATUS.md) and [final deliverables](../deliverables/final/README.md).
> The 10,000-example/full-test plan below was not the executed protocol. Claims
> below about outstanding runs describe the pre-execution planning stage.

# Completing the project before preparing the presentation

The current evidence remains the seed-42 midterm pilot. The final workflow below is
implementation, not evidence that further experiments or participant ratings exist.
The instructor recommends prioritizing placement for this two-person team; the
other P08 objectives remain documented requirements without an explicit waiver.

## Final study design

`experiments/final_protocol.json` defines a new study, separate from the pilot:

- BART-base and both dataset revisions pinned to immutable SHAs.
- Three isolated Q/V placements at rank 8, each with 147,456 trainable parameters.
- Seeds 42, 123, 999; fixed data-selection seed 42.
- 10,000 training examples, three epochs, batch 2 × accumulation 16 = 32.
  Evaluation batch size is 2. Smaller device batches reduce activation memory;
  the same sizes apply to every configuration. Memory fit still needs a real run.
- 512 fixed validation examples, reducing repeated generation cost. These are
  explicitly a subset; final evaluation uses each entire official test split.
- 100 warmup updates out of approximately 936 total updates, followed by linear
  decay. This replaces the long research default of 500 warmup updates and is a
  predeclared starting schedule, not a tuned or validated optimum.
- Existing optimizer, learning rates, truncation and generation defaults retained.
  Full FT uses 3e-5; adapters use 3e-4. No test-set selection of settings/checkpoints.
- Final epoch checkpoint. Rank ablations hold alpha=16 fixed, so alpha/r varies.
- Unchanged decoding on XSum measures transfer under a fixed generation policy;
  minimum length 56 may be unsuitable for XSum. Any alternative is a separate,
  explicitly described validation-selected evaluation, not a test-tuned replacement.

Review loss histories for convergence and runtime on the first run. Any protocol
change starts a new study and requires all compared runs to be repeated. This is
matched exposure, not a demonstrated equal-wall-time or equal-FLOP experiment.

## Execution

Install the README's pinned environment first. Run from the repository root.
The default command only prints a plan and does not download models or train:

```bash
python -m scripts.run_final
```

On a CUDA machine, run stages in order. Each command can be repeated: completed
runs are checked and skipped; incomplete training/evaluations stop with an
explicit message. Keep only one launcher active per study directory.

```bash
python -m scripts.run_final --stage placements --execute
python -m scripts.run_final --stage baselines --execute
python -m scripts.run_final --stage ablations --execute
```

To run the whole predefined matrix in one command:

```bash
python -m scripts.run_final --stage all --execute
```

For shorter sessions, select a mode/seed and optionally train before testing:

```bash
python -m scripts.run_final --stage placements --mode cross_attn --seed 42 --phase train --execute
python -m scripts.run_final --stage placements --mode cross_attn --seed 42 --phase evaluate --execute
```

Use `--output_dir /content/drive/MyDrive/CISC473_P08/final_v1` in Colab. The new
runner must be present in the Colab checkout; the old notebooks pin an earlier code
revision and do not automatically include it. Use the final code snapshot rather
than silently running the old pilot. Pin that snapshot for the entire study.

The stage counts are 12 placement/zero-shot runs, 9 baseline runs and 18 additional
ablation runs. Zero-shot is deterministic; its repeated evaluations are controls,
not independent trained models. If time is limited, complete placements first.
The entire matrix may be expensive, especially full-test decoding. No GPU-duration
or CPU-duration guarantee has been established.

### CPU execution when a GPU is unavailable

CPU execution is supported, but a complete study is likely to require substantial
time and memory. Do not mix CPU FP32 and GPU FP16 into the same comparison.
Create a separate protocol and study directory:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('experiments/final_protocol.json').read_text())
p.update(study='final_cpu_v1', fp16=False)
Path('experiments/final_protocol_cpu.json').write_text(json.dumps(p, indent=2) + '\n')
PY
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.run_final \
  --protocol experiments/final_protocol_cpu.json --output_dir results/final_cpu_v1
```

Add `--execute` only when ready for actual training. Run the existing tiny CPU
tests first; these validate functionality but are not research experiments.
Calibrate an explicitly separate real-model smoke run before committing machine
time to the matrix. All parameters, CPU thread settings and precision should
remain consistent across comparisons. The full model/data must be downloaded.

### Interruptions and provenance

Every study locks its protocol, implementation digest and environment in
`study.json`. Changes require a new output directory. A concurrent launcher fails
on `.launcher.lock`. If a process is forcibly killed, confirm no launcher remains
before removing that stale lock. Do not remove another session's lock.

Partial runs are preserved, not resumed mid-epoch. Archive the incomplete run
outside the study directory before retrying. If training completed and only an
evaluation failed, archive only that incomplete evaluation directory. Keep all
completed run folders with predictions, Trainer state, model/adapter and tokenizer.

## Checking and exporting real results

```bash
python -m scripts.check_final_results --results_dir results/final_v1 \
  --stage all --rescore --output_dir results/final_checks
```

The check expects each configuration/seed and both test domains, verifies saved
configurations and artifacts, validates comparison conditions and capacity, and
recomputes ROUGE from predictions with the official metric. Missing/invalid runs
produce a nonzero exit and a precise `readiness.json`; no final summary is written.
For progress on the main question, use `--stage placements`. Passing that narrower
check does not imply the baseline/ablation matrix or human study is complete.
Use the same `--protocol` when checking a CPU or revised study.

After a complete check:

```bash
python -m src.plot_results --summary results/final_checks/test_summary.json \
  --dataset cnn_dailymail --output results/final_checks/cnn_parameters.png
python -m src.plot_results --summary results/final_checks/test_summary.json \
  --dataset xsum --output results/final_checks/xsum_parameters.png
```

Report ROUGE-1/2/L mean ± sample SD, sample counts, exact trainable counts and
timing scope. Preserve original predictions for independent rescoring, error
analysis and the human study. Do not compare `rougeL` numerically to paper scores
using different metrics/models/protocols as though they were reproduced baselines.

## Remaining work requiring actual evidence

- Collect final experiments under one protocol, including the original seed 42
  again; the 32-update pilot cannot be pooled with longer training.
- Conduct the guide's human study (10 participants, 20 summaries); use the
  [human-evaluation preparation workflow](HUMAN_EVALUATION.md) and retain anonymized ratings.
- Explain failures using source text and generated outputs, without selecting only
  favorable examples. Interpret uncertainty and negative results honestly.
- Separate equal-capacity placement findings from capacity-varying baselines and
  ablations. Measure optimization time separately before making compute claims.
- Finalize the academic report from checked results, actual contributions and
  verified references; disclose AI assistance. No absent score or human judgment
  may be filled in by inference.
- Then prepare the 18–22 minute recording and slides under the presentation handout.

Until those evidence requirements are met or explicitly scoped with the instructor,
the project remains a tested implementation plus a pilot, not a completed final study.
