#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=loc_smoke
#SBATCH --output=results/locality_sensitivity/smoke_gate_%j.out
#SBATCH --error=results/locality_sensitivity/smoke_gate_%j.err
set -euo pipefail
# ---------------------------------------------------------------------------
# H200 fail-fast gate
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU_NAME" != *"H200"* ]]; then
  echo "ERROR: Expected H200, but found: $GPU_NAME" >&2
  exit 2
fi
echo "GPU: $GPU_NAME"
# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load miniconda3/26.1.1
module load cuda/12.6
eval "$(conda shell.bash hook)"
conda activate gpu-byteplane-scan
ROOT_DIR="/home/$USER/workspace/gpu-byteplane-scan-experiments"
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 2
fi
cd "$ROOT_DIR"
# Ensure nvcc is in path
which nvcc
nvcc --version | head -2
# ---------------------------------------------------------------------------
# Build targets on compute node (H200 needs arch 90)
# ---------------------------------------------------------------------------
BUILD_DIR="build_exp4"
cmake -S benchmarks/experiment4 -B "$BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build "$BUILD_DIR" --target byteplane_count_gt_cli -j$(nproc) 2>&1 | tail -5
cmake --build "$BUILD_DIR" --target byteplane_count_gt_batch_cli -j$(nproc) 2>&1 | tail -5
# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
RESULT_DIR="results/locality_sensitivity/smoke_gate_${SLURM_JOB_ID:-local}"
mkdir -p "$RESULT_DIR"
CLI="$ROOT_DIR/build_exp4/byteplane_count_gt_cli"
BATCH_CLI="$ROOT_DIR/build_exp4/byteplane_count_gt_batch_cli"
LOCALITY_ROOT="$ROOT_DIR/datasets/locality_sensitivity"
HEAVY_RAW="/work/$USER/datasets/synthetic/dev/heavy_tailed.f64le.bin"
# heavy_tailed 50% threshold
THRESHOLD="0.9592570420373461"
echo "=== Smoke Gate: Representation Locality Sensitivity ==="
echo "GPU: $GPU_NAME"
echo "Threshold (50%): $THRESHOLD"
echo ""
PASS=0
FAIL=0
for SEG in seg4096 seg16384 seg65536 seg_global; do
  ARTIFACT="$LOCALITY_ROOT/heavy_tailed/$SEG"
  echo "--- Testing: heavy_tailed/$SEG ---"
  if [[ ! -d "$ARTIFACT" ]]; then
    echo "FAIL: artifact dir missing: $ARTIFACT"
    FAIL=$((FAIL + 1))
    continue
  fi
  # Read max_plane_count from manifest
  MAX_K=$(python3 -c "
import json
with open('${ARTIFACT}/manifest.json') as f:
    m = json.load(f)
print(m['max_plane_count'])
")
  echo "  max_plane_count=$MAX_K"
  for K in 1 2 $MAX_K; do
    echo "  k=$K ..."
    # CPU fallback
    CPU_OUT="$RESULT_DIR/heavy_${SEG}_k${K}_cpu.json"
    $CLI \
      --artifact-root "$ARTIFACT" \
      --raw-root "$HEAVY_RAW" \
      --threshold "$THRESHOLD" \
      --k "$K" \
      --no-gpu \
      > "$CPU_OUT" 2>&1
    RC=$?
    echo "    CPU k=$K exit=$RC"
    # GPU
    GPU_OUT="$RESULT_DIR/heavy_${SEG}_k${K}_gpu.json"
    $CLI \
      --artifact-root "$ARTIFACT" \
      --raw-root "$HEAVY_RAW" \
      --threshold "$THRESHOLD" \
      --k "$K" \
      --device 0 \
      > "$GPU_OUT" 2>&1
    RC=$?
    echo "    GPU k=$K exit=$RC"
    # Compare CPU vs GPU
    if [[ -f "$CPU_OUT" ]] && [[ -f "$GPU_OUT" ]]; then
      MISMATCH=$(python3 -c "
import json
cpu = json.load(open('$CPU_OUT'))
gpu = json.load(open('$GPU_OUT'))
fields = ['dataset','rows','threshold','k','max_planes','count','Q','D','U']
bad = [f for f in fields if cpu.get(f) != gpu.get(f)]
if bad:
    print(','.join(bad))
else:
    print('OK')
" 2>&1)
      if [[ "$MISMATCH" == "OK" ]]; then
        GPU_Q=$(python3 -c "import json; print(json.load(open('$GPU_OUT'))['Q'])")
        GPU_D=$(python3 -c "import json; print(json.load(open('$GPU_OUT'))['D'])")
        GPU_U=$(python3 -c "import json; print(json.load(open('$GPU_OUT'))['U'])")
        PREP_MS=$(python3 -c "import json; print(json.load(open('$GPU_OUT'))['threshold_prep_ms'])")
        SCAN_MS=$(python3 -c "import json; print(json.load(open('$GPU_OUT'))['gpu_total_ms'])")
        echo "    PASS: Q=$GPU_Q D=$GPU_D U=$GPU_U prep_ms=$PREP_MS scan_ms=$SCAN_MS"
        PASS=$((PASS + 1))
      else
        echo "    FAIL: mismatch fields: $MISMATCH"
        CPU_CONTENT=$(python3 -c "import json; d=json.load(open('$CPU_OUT')); print({k:d[k] for k in ['count','Q','D','U']})")
        GPU_CONTENT=$(python3 -c "import json; d=json.load(open('$GPU_OUT')); print({k:d[k] for k in ['count','Q','D','U']})")
        echo "    CPU: $CPU_CONTENT"
        echo "    GPU: $GPU_CONTENT"
        FAIL=$((FAIL + 1))
      fi
    else
      echo "    FAIL: output files missing"
      FAIL=$((FAIL + 1))
    fi
  done
  echo ""
done
echo "========================"
echo "Smoke Gate Results: PASS=$PASS FAIL=$FAIL"
echo "========================"
# Write smoke gate CSV
SMOKE_CSV="results/locality_sensitivity/smoke/2026-05-21_Locality_Smoke_Gate.csv"
python3 -c "
import csv, json, os
rows = []
for seg in ['seg4096','seg16384','seg65536','seg_global']:
    for k_str in ['1','2','8']:
        kf_gpu = '$RESULT_DIR/heavy_' + seg + '_k' + k_str + '_gpu.json'
        kf_cpu = '$RESULT_DIR/heavy_' + seg + '_k' + k_str + '_cpu.json'
        if os.path.exists(kf_gpu) and os.path.exists(kf_cpu):
            g = json.load(open(kf_gpu))
            c = json.load(open(kf_cpu))
            match = 'PASS' if g.get('count') == c.get('count') else 'FAIL'
            rows.append({
                'representation': seg,
                'k': k_str,
                'gpu_count': str(g.get('count','?')),
                'cpu_count': str(c.get('count','?')),
                'gpu_Q': str(g.get('Q','?')),
                'gpu_D': str(g.get('D','?')),
                'gpu_U': str(g.get('U','?')),
                'match': match,
                'prep_ms': str(g.get('threshold_prep_ms','?')),
                'scan_ms': str(g.get('gpu_total_ms','?'))
            })
        else:
            rows.append({
                'representation': seg,
                'k': k_str,
                'gpu_count': '?',
                'cpu_count': '?',
                'gpu_Q': '?',
                'gpu_D': '?',
                'gpu_U': '?',
                'match': 'NO_OUTPUT',
                'prep_ms': '?',
                'scan_ms': '?'
            })
with open('$SMOKE_CSV', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['representation','k','gpu_count','cpu_count','gpu_Q','gpu_D','gpu_U','match','prep_ms','scan_ms'])
    w.writeheader()
    w.writerows(rows)
print(f'Smoke CSV written: $SMOKE_CSV ({len(rows)} rows)')
"
echo ""
echo "=== Smoke gate complete ==="
mkdir -p handoff/job_done
MARKER="handoff/job_done/job_${SLURM_JOB_ID:-local}.json"
cat > "$MARKER" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "job_name": "loc_smoke",
  "exit_status": $FAIL,
  "finished_at": "$(date -Iseconds)",
  "workdir": "$ROOT_DIR",
  "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "result_dir": "$RESULT_DIR",
  "next_action": "if smoke passes then run full sweep matrix"
}
EOF
echo "Handoff marker: $MARKER"
exit $FAIL
