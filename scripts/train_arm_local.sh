#!/bin/bash
# Train one arm (CenterDetect -> KeypointDetect -> HybridNet) locally, from
# scratch (random init). Usage: train_arm_local.sh <project> <gpu_id>
# Handles the local env quirks: conda env + libstdc++ LD_PRELOAD.
set -euo pipefail

PROJ="${1:?usage: train_arm_local.sh <project> <gpu_id>}"
GPU="${2:?usage: train_arm_local.sh <project> <gpu_id>}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jarvis
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export CUDA_VISIBLE_DEVICES="$GPU"

REPO=/home/eabe/Research/MyRepos/JARVIS-HybridNet
cd "$REPO"
RUN=fromscratch
LOG="$REPO/projects/$PROJ/train_local.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] === $PROJ on GPU $GPU : CenterDetect ===" | tee -a "$LOG"
python tools/run_train.py CenterDetect   "$PROJ" --pretrain None --run-name "$RUN" 2>&1 | tee -a "$LOG"

echo "[$(date)] === $PROJ on GPU $GPU : KeypointDetect ===" | tee -a "$LOG"
python tools/run_train.py KeypointDetect "$PROJ" --pretrain None --run-name "$RUN" 2>&1 | tee -a "$LOG"

echo "[$(date)] === $PROJ on GPU $GPU : HybridNet ===" | tee -a "$LOG"
python tools/run_train.py HybridNet      "$PROJ" --weights-kp latest --pretrain None --run-name "$RUN" 2>&1 | tee -a "$LOG"

echo "[$(date)] === $PROJ DONE (all 3 stages) ===" | tee -a "$LOG"
