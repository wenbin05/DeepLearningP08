#!/usr/bin/env bash
# One command for the complete midterm run matrix. Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0
output=${OUTPUT_DIR:-results/midterm}
# Resolve once so every independent run uses the same immutable artifacts.
model_revision=$(python -c 'from src.data import resolve_revision; print(resolve_revision("facebook/bart-base", "main"))')
cnn_revision=$(python -c 'from src.data import resolve_revision; print(resolve_revision("abisee/cnn_dailymail", "main", "dataset"))')
xsum_revision=$(python -c 'from src.data import resolve_revision; print(resolve_revision("EdinburghNLP/xsum", "main", "dataset"))')
run() {
    local mode=$1 rank=$2 projections=$3 seed=$4
    local run_dir="$output/${mode}_r${rank}_${projections}_seed${seed}"
    python -m src.train --placement "$mode" --lora_rank "$rank" --projections "$projections" \
        --seed "$seed" --output_dir "$output" --model_revision "$model_revision" \
        --dataset_revision "$cnn_revision" "${@:5}"
    python -m src.train --eval_only "$run_dir" --eval_dataset cnn_dailymail --eval_revision "$cnn_revision"
    python -m src.train --eval_only "$run_dir" --eval_dataset xsum --eval_revision "$xsum_revision"
}
for seed in 42 123 999; do
    for mode in zero_shot full_ft prefix all_attn encoder_self_attn decoder_self_attn cross_attn; do
        run "$mode" 8 q_proj,v_proj "$seed" "$@"
    done
    if [[ ${EXTENDED_ABLATIONS:-0} == 1 ]]; then
        for rank in 4 16; do run all_attn "$rank" q_proj,v_proj "$seed" "$@"; done
        for projections in q_proj k_proj v_proj q_proj,k_proj,v_proj,out_proj; do
            run all_attn 8 "$projections" "$seed" "$@"
        done
    fi
done
python -m src.aggregate_results --results_dir "$output" --output "$output/summary.json"
python -m src.plot_results --summary "$output/summary.json" --output "$output/cnn_pareto.png"
python -m src.plot_results --summary "$output/summary.json" --dataset xsum --output "$output/xsum_pareto.png"
