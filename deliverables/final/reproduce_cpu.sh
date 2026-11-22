#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
main_results=results/cpu_reproduction_v1
full_results=results/cpu_full_ft_reproduction_v1
python -m scripts.run_final --protocol experiments/cpu_placement_protocol.json --output_dir "$main_results" --stage placements --execute
for method in all_attn prefix; do
  python -m scripts.run_final --protocol experiments/cpu_placement_protocol.json --output_dir "$main_results" --stage baselines --mode "$method" --execute
done
python -m scripts.run_final --protocol experiments/cpu_placement_protocol.json --output_dir "$main_results" --stage ablations --execute
MALLOC_ARENA_MAX=2 python -m scripts.run_final --protocol experiments/cpu_full_ft_protocol.json --output_dir "$full_results" --stage baselines --mode full_ft --execute
python deliverables/final/verify_cpu_results.py --main-results "$main_results" --full-results "$full_results"
