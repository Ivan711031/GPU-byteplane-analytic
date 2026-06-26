#!/usr/bin/env bash
#SBATCH --job-name=review_exp3_uci_h200_smoke
#SBATCH --partition=ngstest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/review_exp3_uci_h200_smoke_%j.log
#SBATCH --error=logs/review_exp3_uci_h200_smoke_%j.err
#
# Review/collector for Exp3 UCI H200 smoke jobs.
# Reads the main job's marker, supplements with sacct, inspects CSVs.
# Designed to run after the main smoke job finishes.
#
# Usage:
#   sbatch --export=ALL,MAIN_JOB_ID=<id> scripts/review_exp3_uci_h200_smoke.sbatch
#
# Exits 0 always — this is a collector, not a gate.

set -euo pipefail

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# ---------------------------------------------------------------------------
# Required env
# ---------------------------------------------------------------------------
MAIN_JOB_ID="${MAIN_JOB_ID:-}"
if [[ -z "$MAIN_JOB_ID" ]]; then
  echo "ERROR: MAIN_JOB_ID not set. Export via --export=ALL,MAIN_JOB_ID=<id>" >&2
  exit 0
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JOB_ID="${SLURM_JOB_ID:-nojob}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TIMESTAMP_SHORT="$(date +%Y%m%d_%H%M%S)"

MARKER_DIR="${ROOT_DIR}/handoff/job_done"
MARKER_FILE="${MARKER_DIR}/job_${MAIN_JOB_ID}.json"
REVIEW_MD="${MARKER_DIR}/review_${MAIN_JOB_ID}.md"
REVIEW_JSON="${MARKER_DIR}/review_${MAIN_JOB_ID}.json"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "$MARKER_DIR"

# ---------------------------------------------------------------------------
# Helper: read a JSON string field via Python stdlib
# ---------------------------------------------------------------------------
json_str_field() {
  python3 -c "
import json, sys
try:
    with open('$1') as f:
        d = json.load(f)
    v = d.get('$2', '')
    if v is None:
        v = ''
    print(v)
except Exception:
    print('')
"
}

# ---------------------------------------------------------------------------
# Helper: resolve a path (absolute stays; relative is prefixed with ROOT_DIR)
# ---------------------------------------------------------------------------
resolve_path() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    echo "$p"
  else
    echo "${ROOT_DIR}/${p}"
  fi
}

# ---------------------------------------------------------------------------
# Helper: file size or "missing"
# ---------------------------------------------------------------------------
file_info() {
  local p="$1"
  if [[ -f "$p" ]]; then
    local sz
    sz="$(stat -c%s "$p" 2>/dev/null || echo 0)"
    echo "exists (${sz} bytes)"
  else
    echo "MISSING"
  fi
}

# ===========================================================================
# 1. Marker
# ===========================================================================
echo "[$(date)] reading marker: ${MARKER_FILE}"

MARKER_PRESENT=false
RESULT_DIR=""
RESULT_CSV=""
STDOUT_PATH=""
STDERR_PATH=""
GIT_COMMIT=""
JOB_NAME=""
EXIT_STATUS=""

if [[ -f "$MARKER_FILE" ]]; then
  MARKER_PRESENT=true
  RESULT_DIR="$(json_str_field "$MARKER_FILE" result_dir)"
  RESULT_CSV="$(json_str_field "$MARKER_FILE" result_csv)"
  STDOUT_PATH="$(json_str_field "$MARKER_FILE" stdout)"
  STDERR_PATH="$(json_str_field "$MARKER_FILE" stderr)"
  GIT_COMMIT="$(json_str_field "$MARKER_FILE" git_commit)"
  JOB_NAME="$(json_str_field "$MARKER_FILE" job_name)"
  EXIT_STATUS="$(json_str_field "$MARKER_FILE" exit_status)"
  echo "[$(date)] marker present: job_name=${JOB_NAME} exit_status=${EXIT_STATUS}"
else
  echo "[$(date)] marker NOT found"
fi

# ===========================================================================
# 2. sacct summary
# ===========================================================================
echo "[$(date)] collecting sacct data for job ${MAIN_JOB_ID}"

SACCT_OUTPUT=""
SACCT_MAIN_STATE=""
SACCT_MAIN_EXIT=""
SACCT_MAIN_ELAPSED=""
SACCT_MAIN_NODE=""
SACCT_MAIN_PARTITION=""
SACCT_BATCH_MAXRSS=""

if command -v sacct >/dev/null 2>&1; then
  SACCT_OUTPUT="$(sacct --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Partition,MaxRSS \
    --parsable2 --noheader -j "$MAIN_JOB_ID" 2>/dev/null || true)"

  if [[ -n "$SACCT_OUTPUT" ]]; then
    while IFS='|' read -r sacct_jobid sacct_jobname sacct_state sacct_exit sacct_elapsed sacct_node sacct_part sacct_rss; do
      # Only match the exact job ID (not .batch or .extern steps)
      if [[ "$sacct_jobid" == "$MAIN_JOB_ID" ]]; then
        SACCT_MAIN_STATE="$sacct_state"
        SACCT_MAIN_EXIT="$sacct_exit"
        SACCT_MAIN_ELAPSED="$sacct_elapsed"
        SACCT_MAIN_NODE="$sacct_node"
        SACCT_MAIN_PARTITION="$sacct_part"
      fi
      # Also capture .batch step RSS
      if [[ "$sacct_jobid" == "${MAIN_JOB_ID}.batch" ]]; then
        SACCT_BATCH_MAXRSS="$sacct_rss"
      fi
    done <<< "$SACCT_OUTPUT"
  fi
  echo "[$(date)] sacct: state=${SACCT_MAIN_STATE:-unknown} exit=${SACCT_MAIN_EXIT:-unknown} elapsed=${SACCT_MAIN_ELAPSED:-unknown} node=${SACCT_MAIN_NODE:-unknown}"
else
  echo "[$(date)] sacct not available"
fi

# ===========================================================================
# 3. Log files
# ===========================================================================
echo "[$(date)] checking log files"

STDOUT_ABS=""
STDERR_ABS=""
STDOUT_INFO=""
STDERR_INFO=""

if [[ -n "$STDOUT_PATH" ]]; then
  STDOUT_ABS="$(resolve_path "$STDOUT_PATH")"
  STDOUT_INFO="$(file_info "$STDOUT_ABS")"
fi
if [[ -n "$STDERR_PATH" ]]; then
  STDERR_ABS="$(resolve_path "$STDERR_PATH")"
  STDERR_INFO="$(file_info "$STDERR_ABS")"
fi

echo "[$(date)] stdout (${STDOUT_ABS}): ${STDOUT_INFO}"
echo "[$(date)] stderr (${STDERR_ABS}): ${STDERR_INFO}"

# ===========================================================================
# 4. Result CSV
# ===========================================================================
echo "[$(date)] checking result CSV"

CSV_ABS=""
CSV_INFO=""
CSV_HEADER=""
CSV_FIRST_ROW=""
CSV_LAST_ROW=""
CSV_ROW_COUNT=0

if [[ -n "$RESULT_CSV" ]]; then
  CSV_ABS="$(resolve_path "$RESULT_CSV")"
  if [[ -f "$CSV_ABS" ]]; then
    CSV_ROW_COUNT="$(wc -l < "$CSV_ABS" 2>/dev/null || echo 0)"
    CSV_INFO="exists (${CSV_ROW_COUNT} rows)"
    if (( CSV_ROW_COUNT > 0 )); then
      CSV_HEADER="$(head -1 "$CSV_ABS" 2>/dev/null || true)"
      if (( CSV_ROW_COUNT >= 2 )); then
        CSV_FIRST_ROW="$(head -2 "$CSV_ABS" | tail -1 2>/dev/null || true)"
      fi
      CSV_LAST_ROW="$(tail -1 "$CSV_ABS" 2>/dev/null || true)"
    fi
  else
    CSV_INFO="MISSING"
  fi
fi
echo "[$(date)] csv (${CSV_ABS}): ${CSV_INFO}"

# ===========================================================================
# 5. Write Markdown review
# ===========================================================================
echo "[$(date)] writing ${REVIEW_MD}"

{
  echo "# Review: Exp3 UCI H200 Smoke — Job ${MAIN_JOB_ID}"
  echo ""
  echo "**Generated:** ${TIMESTAMP}"
  echo "**Review job ID:** ${JOB_ID}"
  echo ""
  echo "---"
  echo ""
  echo "## Marker"
  echo ""
  echo "- **Present:** ${MARKER_PRESENT}"
  if $MARKER_PRESENT; then
    echo "- **Job name:** ${JOB_NAME}"
    echo "- **Exit status:** ${EXIT_STATUS}"
    echo "- **Git commit:** ${GIT_COMMIT}"
    echo "- **File:** \`${MARKER_FILE}\`"
  else
    echo "- **Expected at:** \`${MARKER_FILE}\`"
    echo ""
    echo "> The main job did **not** produce a completion marker. This typically means"
    echo "> the job failed before the \`trap\` handler could write it (e.g., \`mkdir -p\`"
    echo "> failed due to wrong \`ROOT_DIR\`, or \`module purge\` failed before trap was"
    echo "> installed in the original script). Check Slurm stderr / \`sacct\` for details."
  fi
  echo ""
  echo "## sacct Summary"
  echo ""
  if [[ -n "$SACCT_MAIN_STATE" ]]; then
    echo "| Field | Value |"
    echo "|-------|-------|"
    echo "| **State** | ${SACCT_MAIN_STATE} |"
    echo "| **ExitCode** | ${SACCT_MAIN_EXIT} |"
    echo "| **Elapsed** | ${SACCT_MAIN_ELAPSED} |"
    echo "| **Node** | ${SACCT_MAIN_NODE} |"
    echo "| **Partition** | ${SACCT_MAIN_PARTITION} |"
    echo "| **MaxRSS (.batch)** | ${SACCT_BATCH_MAXRSS} |"
  else
    echo "No sacct data available (job may have been purged from accounting)."
  fi
  if [[ -n "$SACCT_OUTPUT" ]]; then
    echo ""
    echo "### Raw sacct"
    echo "\`\`\`"
    echo "$SACCT_OUTPUT"
    echo "\`\`\`"
  fi
  echo ""
  echo "## Log Files"
  echo ""
  if [[ -n "$STDOUT_ABS" ]]; then
    echo "- **stdout:** \`${STDOUT_ABS}\` — ${STDOUT_INFO}"
  else
    echo "- **stdout:** not recorded in marker"
  fi
  if [[ -n "$STDERR_ABS" ]]; then
    echo "- **stderr:** \`${STDERR_ABS}\` — ${STDERR_INFO}"
  else
    echo "- **stderr:** not recorded in marker"
  fi
  echo ""
  echo "## Result CSV"
  echo ""
  if [[ -n "$CSV_ABS" ]]; then
    echo "- **Path:** \`${CSV_ABS}\`"
    echo "- **Status:** ${CSV_INFO}"
    if [[ -n "$CSV_HEADER" ]]; then
      echo ""
      echo "### Header (columns)"
      echo "\`\`\`"
      echo "${CSV_HEADER}"
      echo "\`\`\`"
    fi
    if [[ -n "$CSV_FIRST_ROW" ]]; then
      echo ""
      echo "### First data row"
      echo "\`\`\`"
      echo "${CSV_FIRST_ROW}"
      echo "\`\`\`"
    fi
    if [[ -n "$CSV_LAST_ROW" ]] && [[ "$CSV_FIRST_ROW" != "$CSV_LAST_ROW" ]]; then
      echo ""
      echo "### Last data row"
      echo "\`\`\`"
      echo "${CSV_LAST_ROW}"
      echo "\`\`\`"
    fi
    if (( CSV_ROW_COUNT >= 3 )); then
      echo ""
      echo "**Total data rows:** $(( CSV_ROW_COUNT - 1 )) (${CSV_ROW_COUNT} lines including header)"
    fi
  else
    echo "- **Path:** not recorded in marker"
  fi
  echo ""
  echo "---"
  echo ""
  echo "*Review generated by job ${JOB_ID}*"
} > "$REVIEW_MD"

echo "[$(date)] markdown written: ${REVIEW_MD}"

# ===========================================================================
# 6. Write JSON review
# ===========================================================================
echo "[$(date)] writing ${REVIEW_JSON}"

python3 -c "
import json

review = {
    'review_job_id': '${JOB_ID}',
    'main_job_id': '${MAIN_JOB_ID}',
    'generated_at': '${TIMESTAMP}',
    'marker_present': ${MARKER_PRESENT},
    'marker_file': '${MARKER_FILE}',
    'sacct': {
        'state': '${SACCT_MAIN_STATE}',
        'exit_code': '${SACCT_MAIN_EXIT}',
        'elapsed': '${SACCT_MAIN_ELAPSED}',
        'node': '${SACCT_MAIN_NODE}',
        'partition': '${SACCT_MAIN_PARTITION}',
        'batch_maxrss': '${SACCT_BATCH_MAXRSS}'
    },
    'logs': {
        'stdout_path': '${STDOUT_ABS}',
        'stdout_info': '${STDOUT_INFO}',
        'stderr_path': '${STDERR_ABS}',
        'stderr_info': '${STDERR_INFO}'
    },
    'result_csv': {
        'path': '${CSV_ABS}',
        'info': '${CSV_INFO}',
        'row_count': ${CSV_ROW_COUNT}
    }
}

with open('${REVIEW_JSON}', 'w') as f:
    json.dump(review, f, indent=2)
    f.write(chr(10))
"

echo "[$(date)] JSON written: ${REVIEW_JSON}"

# ===========================================================================
# 7. Summary line
# ===========================================================================
echo ""
echo "=== Review complete ==="
echo "  Markdown: ${REVIEW_MD}"
echo "  JSON:     ${REVIEW_JSON}"
