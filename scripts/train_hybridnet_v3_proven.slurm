#!/bin/bash
# Retrain ONLY the HybridNet (3D) stage for unified_V3_masked with V2-matched,
# proven hyperparameters after diagnosing that the first V3 HybridNet collapsed
# the 3D keypoints toward the body center (2D KeypointDetect was fine).
#
# Config now set to: BATCH_SIZE 1, MAX_LEARNING_RATE 0.003, LAPLACIAN_WEIGHT 0.0
# (matches the working unified_V2_masked HybridNet). KeypointDetect weights are
# kept (--weights_keypoint_detect latest -> the good V3 KP run). 3D_only mode.
#
# Submit with:  sbatch scripts/train_hybridnet_v3_proven.slurm
#SBATCH --job-name=v3_hybfix_train
#SBATCH --partition=gpu-l40s
#SBATCH --account=portia
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-v3hybfix-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
set -x
source ~/.bashrc
micromamba activate jarvis
unset LD_LIBRARY_PATH
echo "$SLURMD_NODENAME"
nvidia-smi
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
jarvis-local train hybridNet --weights_keypoint_detect latest unified_V3_masked
