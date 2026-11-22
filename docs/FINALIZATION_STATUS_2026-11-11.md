# Finalization status — 11 November 2026

## Completed in this pass

- Cross-configuration validation now checks training conditions, model/environment,
  seeds, decoding and matched placement capacity before plotting comparisons.
- A staged final-study launcher supports explicit execution, verified skips of
  completed runs, protocol/environment/source locking, and preservation of partial
  runs. Batch-size controls support a lower-memory, effective-batch-32 protocol.
- A final-results check requires every requested configuration/seed and both test
  domains, retained artifacts and compatible protocols. Optional independent ROUGE
  rescoring checks prediction counts/order against the recorded example hash.
- Human-evaluation tools prepare blinded packets and blank forms for 10 participants
  and 20 summaries. Analysis rejects incomplete/duplicate judgments and reports
  descriptive agreement and consistency statistics.
- Existing pilot files and academic reports were preserved. Historical audit notes
  now acknowledge the instructor's supplied feedback and two-person context.

## Verification actually performed

- **41 tests passed** in the pinned CPU environment (46 dependency warnings).
  Includes original model/metric/save-reload tests plus invalid-comparison,
  missing-evidence, rescoring, launcher-lock/restart and human-rating checks.
- A real BART-base cross-attention smoke run completed on CPU with 4 training
  examples, 1 optimizer update and 2 validation examples. It saved adapter,
  tokenizer, predictions, Trainer state, manifest and completed results.
- Exported predictions were independently rescored with the official ROUGE metric.
- The smoke training loop took approximately 24.2 seconds including its epoch
  validation/checkpoint work; the final validation pass took 15.8 seconds.
  Downloads and model/data preparation are excluded. These measurements do not
  estimate the full study's duration and are not comparable research results.
- Shell/source checks and a final-launcher dry run completed. No CUDA GPU was
  available, and no Colab execution was performed during this pass.

The real smoke artifacts are under
`results/final_smoke_cpu/cross_attn_r8_q_proj,v_proj_seed42` in the working environment
(ignored by Git). Keep them separate from the final study and historical pilot.
They establish execution functionality, not convergence or placement quality.

## Still required for a completed final project

There are no newer research results beyond the midterm pilot, and the team has
reported no current GPU access. The final matrix has not run. Multi-seed placement
results, baseline/ablation evidence, held-out transfer, genuine participant ratings,
final analysis and the final academic report remain outstanding.

Use [FINALIZATION.md](FINALIZATION.md) for CPU/GPU execution and evidence checks,
and [HUMAN_EVALUATION.md](HUMAN_EVALUATION.md) for participant collection. Full CPU
execution is possible in principle but has not been timed or completed; the tested
small run does not establish that every larger configuration fits in memory.

Do not describe this engineering pass as completion of the final experiments or
substitute smoke outputs for research findings. The paper's final claims must be
written from the genuine final outputs, or explicitly limited to pilot evidence.
