#!/bin/bash
# Submit a SLURM array job to predict all in-scope Session1 recordings with the
# unified_V3_masked model via the SAM3 shard pipeline. One array task per
# recording (manifest line); each task shards its bouts across NUM_GPUS/2 GPUs.
#
# Usage:
#   DRY_RUN=1 ./submit_session1_array.sh     # print the plan (default)
#   DRY_RUN=0 ./submit_session1_array.sh     # actually submit
# Override a subset, e.g. re-run one recording:
#   DRY_RUN=0 ARRAY_SPEC=7 ./submit_session1_array.sh
set -euo pipefail

JARVIS_ROOT=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
PROJECT=${PROJECT:-unified_V3_masked}
NUM_GPUS=${NUM_GPUS:-4}
THROTTLE=${THROTTLE:-5}
PARTITION=${PARTITION:-ckpt-g2}
MEM=${MEM:-128}
CPUS=${CPUS:-16}
TIME_LIMIT=${TIME_LIMIT:-1-00:00:00}
CONDA_ENV=${CONDA_ENV:-jarvis}
DRY_RUN=${DRY_RUN:-1}
MANIFEST=${MANIFEST:-$JARVIS_ROOT/session1_manifest.tsv}

CENTER_W=/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet/projects/red_data_unified/models/CenterDetect/phase4_center_ft/EfficientTrack-medium_final.pth
KP_W=$JARVIS_ROOT/projects/unified_V3_masked/models/KeypointDetect/Run_20260619-185121/EfficientTrack-large_final.pth
HYBRID_W=$JARVIS_ROOT/projects/unified_V3_masked/models/HybridNet/Run_20260620-173554/HybridNet-large_final.pth
PREDICT_SCRIPT=$JARVIS_ROOT/tools/predict3D_multianimal_shard.py
NODELIST="g[3090-3137]"

# --- validate inputs ---
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST (run gen_session1_manifest.sh)" >&2; exit 1; }
for w in "$CENTER_W" "$KP_W" "$HYBRID_W"; do
  [[ -f "$w" ]] || { echo "ERROR: missing weights: $w" >&2; exit 1; }
done
[[ -f "$PREDICT_SCRIPT" ]] || { echo "ERROR: missing predict script: $PREDICT_SCRIPT" >&2; exit 1; }
(( NUM_GPUS % 2 == 0 )) || { echo "ERROR: NUM_GPUS must be even (each shard uses 2)" >&2; exit 1; }

N=$(wc -l < "$MANIFEST")
ARRAY_SPEC=${ARRAY_SPEC:-1-${N}%${THROTTLE}}

mkdir -p "$JARVIS_ROOT/OutFiles"

echo "Project: $PROJECT | partition: $PARTITION | GPUs/task: $NUM_GPUS | array: $ARRAY_SPEC | DRY_RUN=$DRY_RUN"
echo "Manifest: $MANIFEST ($N recordings)"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "=== task -> recording map ==="
  nl -w2 -s'  ' "$MANIFEST" | sed 's#\t# | bouts=#'
  echo
  echo "=== resolved command for array task 1 ==="
  line=$(sed -n '1p' "$MANIFEST"); rec_dir=$(cut -f1 <<<"$line"); bouts_csv=$(cut -f2 <<<"$line")
  cat <<CMD
python -u $PREDICT_SCRIPT \\
    --project $PROJECT --video_folder $rec_dir --calib_folder $rec_dir/calibration \\
    --num_animals 2 --num_gpus $NUM_GPUS --bouts_csv $bouts_csv \\
    --output_name <arrayjob>_1 \\
    --center-weights $CENTER_W --kp-weights $KP_W --hybridnet-weights $HYBRID_W \\
    --save-masks --save-clips
CMD
  echo
  echo "DRY_RUN — nothing submitted. Re-run with DRY_RUN=0 to submit."
  exit 0
fi

sbatch --array=${ARRAY_SPEC} <<EOF
#!/bin/bash
#SBATCH --job-name=s1pred
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
#SBATCH -o ${JARVIS_ROOT}/OutFiles/slurm-session1-%A_%a.out
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

line=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")
rec_dir=\$(cut -f1 <<<"\$line")
bouts_csv=\$(cut -f2 <<<"\$line")
echo "[task \${SLURM_ARRAY_TASK_ID}] rec=\$rec_dir bouts=\$bouts_csv"

cd ${JARVIS_ROOT}
python -u ${PREDICT_SCRIPT} \\
    --project ${PROJECT} \\
    --video_folder \$rec_dir \\
    --calib_folder \$rec_dir/calibration \\
    --num_animals 2 \\
    --num_gpus ${NUM_GPUS} \\
    --bouts_csv \$bouts_csv \\
    --output_name \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID} \\
    --center-weights ${CENTER_W} \\
    --kp-weights ${KP_W} \\
    --hybridnet-weights ${HYBRID_W} \\
    --save-masks \\
    --save-clips
EOF

echo "Submitted array ${ARRAY_SPEC}. Monitor: squeue -u \$USER | Logs: ${JARVIS_ROOT}/OutFiles/slurm-session1-*.out"
