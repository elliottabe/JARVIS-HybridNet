#!/bin/bash
# Train the HybridNet (3D) stage for unified_V3_masked, picking up the latest
# KeypointDetect weights and enabling the graph-Laplacian shape prior
# (HYBRIDNET.LAPLACIAN_WEIGHT in projects/unified_V3_masked/config.yaml).
#
# Mirrors the resources of the completed KeypointDetect job (v3_kpL_train,
# 36213535): gpu-l40s, 1 GPU, 16 CPU, 64G, account portia.
#
# Submit with:  sbatch scripts/train_hybridnet_v3.slurm
#SBATCH --job-name=v3_hyb_train
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
#SBATCH -o ./OutFiles/slurm-v3hyb-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
# 3d_tracking (py3.12) packages need a newer libstdc++ than system /lib64;
# preload the env's own (matches scripts/train_arm_local.sh).
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
echo "$SLURMD_NODENAME"
nvidia-smi
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
jarvis-local train hybridNet --weights_keypoint_detect latest unified_V3_masked
