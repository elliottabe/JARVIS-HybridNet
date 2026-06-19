#!/bin/bash
# Re-predict all courtship bouts with a new JARVIS model (default: unified_V2_masked).
#
# Submits ONE SLURM job per recording, feeding --bouts_csv directly from each
# recording's existing courtship_bouts_unified_summary.csv. This bypasses
# slurm_predict.py's bouts_root auto-mapping, which needs an info.yaml that the
# shard-pipeline output folders don't have.
#
# Usage:
#   DRY_RUN=1 ./submit_courtship_repredict.sh      # show what would be submitted (default)
#   DRY_RUN=0 ./submit_courtship_repredict.sh      # actually submit
#
# Override defaults via env, e.g.:
#   PROJECT=unified_V2_masked NUM_GPUS=4 PARTITION=ckpt-g2 DRY_RUN=0 ./submit_courtship_repredict.sh
set -euo pipefail

PROJECT=${PROJECT:-unified_V2_masked}
NUM_GPUS=${NUM_GPUS:-4}
PARTITION=${PARTITION:-ckpt-g2}
MEM=${MEM:-128}
CPUS=${CPUS:-16}
TIME_LIMIT=${TIME_LIMIT:-1-00:00:00}
CONDA_ENV=${CONDA_ENV:-jarvis}
DRY_RUN=${DRY_RUN:-1}
ONLY_SESSION=${ONLY_SESSION:-}   # e.g. ONLY_SESSION=Session0 to submit just one session

JARVIS_ROOT=/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet
VID_ROOT=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship
# Use the SAM3 video-propagation shard pipeline (the proven courtship path).
# Each shard uses 2 GPUs (JARVIS + SAM3), so NUM_GPUS must be even.
PREDICT_SCRIPT=$JARVIS_ROOT/tools/predict3D_multianimal_shard.py

# Weights for the new masked (4-channel KeypointDetect) model. The shard
# script's defaults point at the old red_data_unified project, so pass these
# explicitly. The trained weights live under each net's fromscratch/ dir.
MODELS=$JARVIS_ROOT/projects/$PROJECT/models
CENTER_W=${CENTER_W:-$MODELS/CenterDetect/fromscratch/EfficientTrack-medium_final.pth}
KP_W=${KP_W:-$MODELS/KeypointDetect/fromscratch/EfficientTrack-medium_final.pth}
HYBRID_W=${HYBRID_W:-$MODELS/HybridNet/fromscratch/HybridNet-medium_final.pth}

# bouts_root dirs that hold the *_bouts_unified_summary.csv files (per session)
BOUTS_ROOTS=(
  "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_04172026:Session0"
  "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session1_bouts_04172026:Session1"
)

# ckpt-g2 nodelist (matches slurm_predict.py)
NODELIST="g[3090-3137]"

mkdir -p "$JARVIS_ROOT/OutFiles"

submit_one() {
  local rec_dir="$1" csv="$2"
  local rec_name; rec_name=$(basename "$rec_dir")
  local calib="$rec_dir/calibration"

  if [[ ! -d "$calib" ]]; then
    echo "  [skip] $rec_name: no calibration/ folder"
    return
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [DRY] $rec_name"
    echo "        video : $rec_dir"
    echo "        bouts : $csv"
    return
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=reprD_${rec_name}
#SBATCH --partition=${PARTITION}
#SBATCH --account=portia
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus=${NUM_GPUS}
#SBATCH --mem=${MEM}G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o ${JARVIS_ROOT}/OutFiles/slurm-reprD-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
#SBATCH --nodelist=${NODELIST}
#SBATCH --exclude=g[3107,3115,3109]
module load cuda/12.9.1
module load gcc/12
set -x
source ~/.bashrc
nvidia-smi
micromamba activate ${CONDA_ENV}
unset LD_LIBRARY_PATH
echo \$SLURMD_NODENAME
cd ${JARVIS_ROOT}
python -u ${PREDICT_SCRIPT} \\
    --project ${PROJECT} \\
    --video_folder ${rec_dir} \\
    --calib_folder ${calib} \\
    --num_animals 2 \\
    --num_gpus ${NUM_GPUS} \\
    --bouts_csv ${csv} \\
    --output_name \$SLURM_JOB_ID \\
    --center-weights ${CENTER_W} \\
    --kp-weights ${KP_W} \\
    --hybridnet-weights ${HYBRID_W} \\
    --save-masks \\
    --save-clips
EOF
}

echo "Project: $PROJECT | partition: $PARTITION | GPUs/job: $NUM_GPUS | DRY_RUN=$DRY_RUN"
echo

n=0
for entry in "${BOUTS_ROOTS[@]}"; do
  broot="${entry%%:*}"
  sess="${entry##*:}"
  if [[ -n "$ONLY_SESSION" && "$sess" != "$ONLY_SESSION" ]]; then
    continue
  fi
  echo "=== $sess  ($broot) ==="
  while IFS= read -r csv; do
    # recording name = parent-of-parent of the CSV (.../<rec>/Predictions_3D_*/csv)
    cand=$(basename "$(dirname "$(dirname "$csv")")")
    rec_dir="$VID_ROOT/$sess/$cand"
    if [[ ! -d "$rec_dir" ]]; then
      # flat layout (e.g. Session0): broot has no per-recording subdir.
      # Fall back to the single recording in this session if unambiguous.
      mapfile -t recs < <(find "$VID_ROOT/$sess" -mindepth 1 -maxdepth 1 -type d)
      if [[ ${#recs[@]} -eq 1 ]]; then
        rec_dir="${recs[0]}"
      else
        echo "  [skip] could not map CSV to a recording: $csv"
        continue
      fi
    fi
    submit_one "$rec_dir" "$csv"
    n=$((n+1))
  done < <(find "$broot" -name courtship_bouts_unified_summary.csv | sort)
  echo
done

echo "Total recordings: $n  (DRY_RUN=$DRY_RUN)"
[[ "$DRY_RUN" == "1" ]] && echo "Re-run with DRY_RUN=0 to actually submit." || \
  echo "Monitor: squeue -u \$USER   |   Outputs land in each recording's Predictions_3D_<jobid>/"
