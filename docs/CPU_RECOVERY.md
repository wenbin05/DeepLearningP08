# CPU run recovery — 13 November 2026

The original full fine-tuning run (seed 42) was killed by the 8 GiB memory limit at step 69 of 96. Its incomplete artifacts, including the step-64 checkpoint, are preserved under `results/interrupted/cpu_placement_v1/`. They are not completed research results.

The 18 completed runs and 36 test evaluations remain intact. The original training source hash was verified unchanged during recovery.

Full fine-tuning restarts from the pretrained model in `results/cpu_full_ft_batch1_v1/`, using `experiments/cpu_full_ft_protocol.json`: microbatch 1, gradient accumulation 32, effective batch 32, and MALLOC_ARENA_MAX=2. All other protocol fields except the study name are unchanged. The two-update execution probe passed (peak child RSS about 4.73 GiB); this short probe does not guarantee that the entire run will fit and is not research evidence.

Report this baseline as memory-adjusted. Equal effective batch size does not imply numerically identical optimization: batching can change loss weighting and dropout behavior. The main equal-capacity LoRA placement comparison retains its original consistent settings.

The recovery supervisor runs the three full fine-tuning seeds, then the 18 adapter ablations. A full fine-tuning failure is recorded and does not prevent launching independent ablations. The full-matrix verifier preserves the batching exception and validates the original and adjusted protocols separately before producing combined summaries. Scripts are archived in `results/recovery_logs/`.

Live state: `results/cpu_placement_v1/status.json` and `recovery_queue.json`. Log: `results/cpu_placement_v1/recovery_execution.log`. Final readiness requires complete artifacts and successful checks; process activity alone is not completion.
