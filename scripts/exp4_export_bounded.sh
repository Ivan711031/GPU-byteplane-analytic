#!/bin/bash
#SBATCH --partition=ngs32g
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm_exp4_export_bounded_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJ_ROOT="/home/u4063895/workspace/gpu-byteplane-scan-experiments"
DATA_ROOT="/work/u4063895/datasets/synthetic/dev"
OUT_ROOT="/work/u4063895/datasets/synthetic"
EXPORT_BIN="${PROJ_ROOT}/build/exp3/export_encoded_dev_layout"

cd "${PROJ_ROOT}"
mkdir -p results

DATASETS=(uniform heavy_tailed sensor zipfian)

echo "=== Exporting precision-decimals=3 => dev_buff_exp4_p3 ===" >&2
mkdir -p "${OUT_ROOT}/dev_buff_exp4_p3"
for ds in "${DATASETS[@]}"; do
    echo "  ${ds} ..." >&2
    "${EXPORT_BIN}" \
        --input "${DATA_ROOT}/${ds}.f64le.bin" \
        --output-root "${OUT_ROOT}/dev_buff_exp4_p3" \
        --precision-decimals 3
done

echo "=== Exporting precision-decimals=6 => dev_buff_exp4_p6 ===" >&2
mkdir -p "${OUT_ROOT}/dev_buff_exp4_p6"
for ds in "${DATASETS[@]}"; do
    echo "  ${ds} ..." >&2
    "${EXPORT_BIN}" \
        --input "${DATA_ROOT}/${ds}.f64le.bin" \
        --output-root "${OUT_ROOT}/dev_buff_exp4_p6" \
        --precision-decimals 6
done

echo "=== All exports complete ===" >&2

# Completion marker
MARKER_DIR="${PROJ_ROOT}/handoff/job_done"
mkdir -p "${MARKER_DIR}"
cat > "${MARKER_DIR}/job_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME:-exp4_bounded_export}",
  "exit_status": 0,
  "finished_at": "$(date -Iseconds)",
  "workdir": "${PROJ_ROOT}",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "stdout": "${PROJ_ROOT}/results/slurm_exp4_export_bounded_${SLURM_JOB_ID}.out",
  "stderr": "same_as_stdout",
  "result_dirs": ["${OUT_ROOT}/dev_buff_exp4_p3", "${OUT_ROOT}/dev_buff_exp4_p6"],
  "next_action": "Build and run Exp4 smoke test on H200 against dev_buff_exp4_p3 and dev_buff_exp4_p6"
}
EOF
