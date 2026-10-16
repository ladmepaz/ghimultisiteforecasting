#!/usr/bin/env bash
# run_all.sh
# Download the data, train every model for India and Colombia, evaluate them and build
# the manuscript's tables and figures.
# Stops at the first step that fails; each step's output is also saved to logs/run_<time>/.
#
# Run from the repo root, with the conda environment activated:
#     bash run_all.sh
# Resume from a later step (e.g. after fixing a failure in step 5):
#     bash run_all.sh 5

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

FROM=${1:-1}
STEPS=(
    "download_data|data/download_nsrdb.py"
    "individual_india|train_individual.py"
    "individual_colombia|train_individual.py --is-colombia"
    "joint_india|train_joint.py"
    "joint_colombia|train_joint.py --is-colombia"
    "gnn_india|train_gnn.py"
    "gnn_colombia|train_gnn.py --is-colombia"
    "evaluate_india|evaluate.py"
    "evaluate_colombia|evaluate.py --is-colombia"
    "significance_india|significance_tests.py"
    "significance_colombia|significance_tests.py --is-colombia"
    "tables|make_results_tables.py"
    "figures|plot_results.py"
    "cloud_climate_india|compute_cloud_climate_stats.py"
    "cloud_climate_colombia|compute_cloud_climate_stats.py --is-colombia"
    "cloud_type_india|plot_cloud_type_heatmap.py"
    "cloud_type_colombia|plot_cloud_type_heatmap.py --is-colombia"
)

LOG_DIR="logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Python: $(command -v python)"
echo "Logs:   $LOG_DIR"
echo

if ! python -c "import tensorflow, spektral, h5py" >/dev/null 2>&1; then
    echo "This Python can't import tensorflow, spektral and h5py. Activate the environment"
    echo "from the README (conda activate ghimultisite) and run again."
    exit 1
fi

RUN_START=$SECONDS
for ((n = FROM; n <= ${#STEPS[@]}; n++)); do
    name=${STEPS[n - 1]%%|*}
    args=${STEPS[n - 1]#*|}
    log=$(printf "%s/%02d_%s.log" "$LOG_DIR" "$n" "$name")
    echo "[$n/${#STEPS[@]}] python $args"
    start=$SECONDS
    # shellcheck disable=SC2086
    if ! python -u $args 2>&1 | tee "$log"; then
        echo
        echo "FAILED at step $n ($name) after $(( (SECONDS - start) / 60 )) min."
        echo "Log: $log"
        echo "After fixing it, resume with: bash run_all.sh $n"
        exit 1
    fi
    echo "  done in $(( (SECONDS - start) / 60 )) min"
    echo
done

echo "All steps finished in $(( (SECONDS - RUN_START) / 3600 )) h $(( (SECONDS - RUN_START) % 3600 / 60 )) min."
