#!/bin/bash
# Qualitative V3 prediction + viz/clips on the Phase-1 diagnosis baseline
# recording (Session0/2025_10_20_13_20_04), SAM3 shard pipeline.
#
# Uses the third_party checkout (has the unified_V3_masked project + latest
# code). V3 only retrained KeypointDetect (mask-containment) and HybridNet
# (graph-Laplacian); CenterDetect is borrowed from red_data_unified (its output
# is NOT used for crop centering in SAM3 mode — crops center on the SAM3 mask
# centroid). Runs in the `jarvis` env (has jarvis-local + sam3).
#
# Submit: sbatch scripts/predict_v3_viz.slurm
#SBATCH --job-name=v3_predviz_S0
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-v3predviz-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
module load cuda/12.9.1
module load gcc/12
set -x
source ~/.bashrc
nvidia-smi
micromamba activate jarvis
unset LD_LIBRARY_PATH
echo "$SLURMD_NODENAME"

JARVIS_ROOT=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
REC=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04
BOUTS_CSV=/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_04172026/Predictions_3D_34662304/courtship_bouts_v3subset.csv

# V3 trained weights (large) + borrowed CenterDetect (medium).
CENTER_W=/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet/projects/red_data_unified/models/CenterDetect/phase4_center_ft/EfficientTrack-medium_final.pth
KP_W=$JARVIS_ROOT/projects/unified_V3_masked/models/KeypointDetect/Run_20260619-185121/EfficientTrack-large_final.pth
HYBRID_W=$JARVIS_ROOT/projects/unified_V3_masked/models/HybridNet/Run_20260620-085937/HybridNet-large_final.pth

cd "$JARVIS_ROOT"
python -u tools/predict3D_multianimal_shard.py \
    --project unified_V3_masked \
    --video_folder "$REC" \
    --calib_folder "$REC/calibration" \
    --num_animals 2 \
    --num_gpus 4 \
    --bouts_csv "$BOUTS_CSV" \
    --output_name "$SLURM_JOB_ID" \
    --center-weights "$CENTER_W" \
    --kp-weights "$KP_W" \
    --hybridnet-weights "$HYBRID_W" \
    --save-masks \
    --save-clips
