#!/bin/bash
#SBATCH --job-name=llm_array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=longgpu
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-2

set -euo pipefail

# Module laden
module load cuda/12.4
module load python/3.11

# Venv aktivieren
source /vol/fob-vol3/nebenf24/siversju/myenv/bin/activate

# Base directory
BASE_DIR="/vol/fob-vol3/nebenf24/siversju/ssi/thesis/final50"

# Skripte + Pfade
SCRIPTS=("ask_mistral/ask_mistral.py" "ask_llama/ask_llama.py" "ask_qwen/ask_qwen.py")

SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
DIR=$(dirname "$SCRIPT")

cd $BASE_DIR/$DIR

echo "🚀 Running $SCRIPT on node $(hostname) with GPU(s): $CUDA_VISIBLE_DEVICES"
python $(basename "$SCRIPT")
