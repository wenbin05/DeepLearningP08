#!/usr/bin/env bash
# Run from repository root with the pinned environment activated.
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
python_bin=${P08_PYTHON:-python}
protocol=experiments/cpu_placement_protocol.json
output=results/cpu_placement_v1
# Train the central comparison before spending time on test-set decoding.
for mode in cross_attn encoder_self_attn decoder_self_attn; do
    "$python_bin" -u -m scripts.run_final --protocol "$protocol" --output_dir "$output" \
        --stage placements --mode "$mode" --phase train --execute
done
"$python_bin" -u -m scripts.run_final --protocol "$protocol" --output_dir "$output" \
    --stage placements --execute
"$python_bin" -u -m scripts.check_final_results --protocol "$protocol" --results_dir "$output" \
    --stage placements --rescore --output_dir "$output/placement_checks"
for dataset in cnn_dailymail xsum; do
    "$python_bin" -u -m src.plot_results --summary "$output/placement_checks/test_summary.json" \
        --dataset "$dataset" --output "$output/placement_checks/${dataset}_parameters.png"
done
"$python_bin" -u - <<'PY'
from pathlib import Path
from scripts.human_evaluation import prepare
root=Path('results/cpu_placement_v1')
paths=[]
for mode in ['zero_shot','encoder_self_attn','decoder_self_attn','cross_attn']:
    paths.extend((root/f'{mode}_r8_q_proj,v_proj_seed42'/'evaluations').glob('cnn_dailymail_*/predictions.jsonl'))
output=root/'human_study'
if not output.exists():
    prepare(paths,output)
print('Placement outputs checked; blinded human-study packets:',output)
PY
# Comparison baselines follow the primary research question.
"$python_bin" -u -m scripts.run_final --protocol "$protocol" --output_dir "$output" \
    --stage baselines --execute
"$python_bin" -u -m scripts.check_final_results --protocol "$protocol" --results_dir "$output" \
    --stage baselines --rescore --output_dir "$output/baseline_checks"
printf '%s\n' 'CPU placement and baseline stages complete. Rank/projection ablations and real participant ratings remain separate.'
