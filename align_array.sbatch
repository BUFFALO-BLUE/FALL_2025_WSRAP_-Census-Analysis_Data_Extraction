#!/bin/bash
#SBATCH --job-name=align_census
#SBATCH --output=/gpfs/homefs1/%u/census_project/logs/align_%A_%a.out
#SBATCH --error=/gpfs/homefs1/%u/census_project/logs/align_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-2067%100

set -euo pipefail

# Scaling note: prior prototype used 106 images; this array targets all 2067.

PROJECT_ROOT="/gpfs/homefs1/$USER/census_project"
CODE_DIR="$PROJECT_ROOT/code"
DATA_RAW="$PROJECT_ROOT/data_raw"
DATA_ALIGNED="$PROJECT_ROOT/data_aligned"
OUTPUTS="$PROJECT_ROOT/outputs"
LOGS="$PROJECT_ROOT/logs"

mkdir -p "$CODE_DIR" "$DATA_RAW" "$DATA_ALIGNED" "$OUTPUTS" "$LOGS"

INPUT_LIST="$CODE_DIR/jpg_list.txt"

if [[ ! -f "$INPUT_LIST" ]]; then
  echo "❌ Missing file list at $INPUT_LIST" >&2
  echo "Create it first:" >&2
  echo "  find \"$DATA_RAW/Research Final ver. Jeremy P\" -type f -name \"*.jpg\" | sort > \"$INPUT_LIST\"" >&2
  exit 1
fi

REF_IMAGE="${REF_IMAGE:-$(head -n 1 "$INPUT_LIST")}"
IMAGE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$INPUT_LIST")

if [[ -z "$IMAGE_PATH" ]]; then
  echo "❌ No image found for task ID ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "[$(date)] Aligning image ${SLURM_ARRAY_TASK_ID}/2067" >&2

python -u "$CODE_DIR/scripts/hpc/align_single_image.py" \
  --input "$IMAGE_PATH" \
  --reference "$REF_IMAGE" \
  --output-dir "$DATA_ALIGNED"
