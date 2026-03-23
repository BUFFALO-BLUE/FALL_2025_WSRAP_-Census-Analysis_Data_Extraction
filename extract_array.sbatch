#!/bin/bash
#SBATCH --job-name=extract_census
#SBATCH --output=/gpfs/homefs1/%u/census_project/logs/extract_%A_%a.out
#SBATCH --error=/gpfs/homefs1/%u/census_project/logs/extract_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-2067%100

set -euo pipefail

# Scaling note: prior prototype used 106 images; this array targets all 2067.

PROJECT_ROOT="/gpfs/homefs1/$USER/census_project"
CODE_DIR="$PROJECT_ROOT/code"
DATA_ALIGNED="$PROJECT_ROOT/data_aligned"
OUTPUTS="$PROJECT_ROOT/outputs"
LOGS="$PROJECT_ROOT/logs"

mkdir -p "$CODE_DIR" "$DATA_ALIGNED" "$OUTPUTS" "$LOGS"

ALIGNED_LIST="$CODE_DIR/aligned_jpg_list.txt"

if [[ ! -f "$ALIGNED_LIST" ]]; then
  echo "❌ Missing aligned list at $ALIGNED_LIST" >&2
  echo "Create it after alignment:" >&2
  echo "  find \"$DATA_ALIGNED\" -type f -name \"*.jpg\" | sort > \"$ALIGNED_LIST\"" >&2
  exit 1
fi

IMAGE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ALIGNED_LIST")

if [[ -z "$IMAGE_PATH" ]]; then
  echo "❌ No image found for task ID ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "[$(date)] Extracting image ${SLURM_ARRAY_TASK_ID}/2067" >&2

python -u "$CODE_DIR/scripts/hpc/extract_single_image.py" \
  --input "$IMAGE_PATH" \
  --output-dir "$OUTPUTS"