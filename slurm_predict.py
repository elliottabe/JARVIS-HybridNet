import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def slurm_submit(script):
    """Submit the SLURM script using sbatch and return the job ID."""
    try:
        output = subprocess.check_output(
            ["sbatch"], input=script, universal_newlines=True
        )
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def _build_recording_to_bouts_map(bouts_root, data_dir, dataset):
    """Walk `bouts_root` for Predictions_3D_* folders and match each one to a
    recording folder under `data_dir` via its info.yaml `recording_path` field.

    Returns a list of (recording_folder, bouts_csv, source_predictions_folder)
    tuples. Folders with missing info.yaml, missing recording_path, missing
    bouts CSV, or a recording_path outside `data_dir` are skipped with a
    printed warning (never crash).
    """
    bouts_root = Path(bouts_root).resolve()
    data_dir_resolved = Path(data_dir).resolve()
    if not bouts_root.is_dir():
        print(f"ERROR: --bouts_root not a directory: {bouts_root}", file=sys.stderr)
        sys.exit(1)

    csv_name = f"{dataset}_bouts_unified_summary.csv"
    mapping = []
    for pred_dir in sorted(bouts_root.glob("Predictions_3D_*")):
        if not pred_dir.is_dir():
            continue
        info_yaml = pred_dir / "info.yaml"
        if not info_yaml.exists():
            print(f"  [skip] {pred_dir.name}: no info.yaml")
            continue
        try:
            with open(info_yaml) as fh:
                info = yaml.safe_load(fh) or {}
        except Exception as e:
            print(f"  [skip] {pred_dir.name}: failed to parse info.yaml ({e})")
            continue
        rec_path = info.get("recording_path")
        if not rec_path:
            print(f"  [skip] {pred_dir.name}: no recording_path in info.yaml")
            continue
        rec_path = Path(os.path.realpath(rec_path))
        try:
            rec_path.relative_to(data_dir_resolved)
        except ValueError:
            print(f"  [skip] {pred_dir.name}: recording_path {rec_path} not under --data_dir")
            continue
        if not rec_path.is_dir():
            print(f"  [skip] {pred_dir.name}: recording_path does not exist: {rec_path}")
            continue
        bouts_csv = pred_dir / csv_name
        if not bouts_csv.exists():
            print(f"  [skip] {pred_dir.name}: no {csv_name}")
            continue
        mapping.append((rec_path, bouts_csv, pred_dir))
    return mapping


def submit_predict(
    video_folder,
    calib_folder,
    conda_env_name,
    project,
    partition,
    job_name,
    num_gpus,
    mem,
    cpus,
    time_limit,
    bouts_csv=None,
    stage_dir=None,
    src_pred_dir=None,
    dataset=None,
    predict_script_name='jarvis_batch_multi_animal.py',
    save_clips=False,
    triangulate=False,
):
    """Construct and submit a SLURM prediction job.

    Uses $SLURM_JOB_ID as the output directory name so that preempted jobs
    (which keep the same job ID on requeue) automatically resume.
    """
    gpu_configs = {
        "gpu-a40": "g[3040-3047,3050-3057,3060-3067,3070-3077]",
        "gpu-a100": "g[3080-3087]",
        "gpu-l40": "g[3090-3099,3115-3119]",
        "gpu-l40s": "g[3100-3114,3120-3124,3133-3137]",
        "gpu-h200": "g[3125-3132]",
        "ckpt-g2": "g[3090-3137]",
    }

    gpu_resource = gpu_configs.get(partition, "")
    nodelist_line = f"#SBATCH --nodelist={gpu_resource}" if gpu_resource else ""

    jarvis_root = os.path.dirname(os.path.abspath(__file__))
    # Resolve the predict script relative to the repo root. Accepts a bare
    # filename (legacy default) or a path like `tools/predict3D_multianimal_shard.py`.
    if os.path.isabs(predict_script_name):
        predict_script = predict_script_name
    else:
        predict_script = os.path.join(jarvis_root, predict_script_name)

    # Derive a short name from the recording folder
    recording_name = os.path.basename(str(video_folder).rstrip("/"))

    bouts_line = f" \\\n    --bouts_csv {bouts_csv}" if bouts_csv else ""
    save_clips_line = " \\\n    --save-clips" if save_clips else ""
    triangulate_line = " \\\n    --triangulate" if triangulate else ""

    # In bouts mode, stage the new JARVIS output into a sibling
    # <bouts_root>_bouts/Predictions_3D_<job_id>/ directory so the downstream
    # 3d_tracking pipeline (slurm_run.py / run_full_pipeline.py) can discover
    # it via its existing rglob('Predictions_3D_*') + '<dataset>_bouts*.csv'
    # logic. We symlink data3D_fly{0,1}.csv from the recording's new
    # Predictions folder and symlink info.yaml + bouts summary CSVs from the
    # source Predictions folder (the one under --bouts_root).
    stage_block = ""
    if stage_dir and src_pred_dir and dataset:
        stage_block = f"""
# --- Option A staging: expose this bouts-mode run to the downstream pipeline
STAGE_DIR={stage_dir}
PRED_DIR={video_folder}/Predictions_3D_$SLURM_JOB_ID
if [ -f "$PRED_DIR/data3D_fly0.csv" ] && [ -f "$PRED_DIR/data3D_fly1.csv" ]; then
    mkdir -p "$STAGE_DIR"
    ln -sfn "$PRED_DIR/data3D_fly0.csv" "$STAGE_DIR/data3D_fly0.csv"
    ln -sfn "$PRED_DIR/data3D_fly1.csv" "$STAGE_DIR/data3D_fly1.csv"
    ln -sfn {src_pred_dir}/info.yaml "$STAGE_DIR/info.yaml"
    for f in {src_pred_dir}/{dataset}_bouts_*summary.csv; do
        [ -f "$f" ] && ln -sfn "$f" "$STAGE_DIR/$(basename $f)"
    done
    [ -f "$PRED_DIR/tracking_info.json" ] && ln -sfn "$PRED_DIR/tracking_info.json" "$STAGE_DIR/tracking_info.json"
    echo "Staged bouts prediction to $STAGE_DIR"
else
    echo "WARNING: expected data3D_fly{{0,1}}.csv missing in $PRED_DIR, skipping staging"
fi
"""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{recording_name}
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={num_gpus}
#SBATCH --mem={mem}G
#SBATCH --requeue
#SBATCH --verbose
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-predict-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
{nodelist_line}
#SBATCH --exclude=g[3107,3115,3109]
module load cuda/12.9.1
module load gcc/12
set -x
source ~/.bashrc
nvidia-smi
micromamba activate {conda_env_name}
unset LD_LIBRARY_PATH
echo $SLURMD_NODENAME
echo "Processing recording: {video_folder}"
python -u {predict_script} \\
    --project {project} \\
    --video_folder {video_folder} \\
    --calib_folder {calib_folder} \\
    --num_animals 2 \\
    --num_gpus {num_gpus} \\
    --output_name $SLURM_JOB_ID{bouts_line}{save_clips_line}{triangulate_line}
{stage_block}"""
    print(f"Submitting: {recording_name}")
    job_id = slurm_submit(script)
    print(f"  Job ID: {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM prediction jobs for JARVIS-HybridNet. "
        "Submits one job per recording folder. Uses SLURM job ID as the "
        "output directory name so preempted jobs auto-resume on requeue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all recordings in Session11:
  python slurm_predict.py --data_dir /gscratch/portia/eabe/data/Johnson_lab/3d_data/Session11

  # Use a specific partition:
  python slurm_predict.py --data_dir /path/to/session --partition gpu-l40s

  # Predict a single recording:
  python slurm_predict.py --data_dir /path/to/session/2026_03_03_13_25_13

  # Re-run into a date-stamped stage dir so it stays separate from an existing one:
  python slurm_predict.py --data_dir /path/to/session \\
      --bouts_root /path/to/session/04092026 \\
      --stage_suffix $(date +%m%d%Y)

  # Dry run (see what would be submitted):
  python slurm_predict.py --data_dir /path/to/session --dry_run
""",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to session directory containing recording folders, "
        "or path to a single recording folder",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="courtship_multianimal_V1",
        help="JARVIS project name (default: courtship_multianimal_V1)",
    )
    parser.add_argument(
        "--conda_env_name",
        type=str,
        default="jarvis",
        help="Conda environment name (default: jarvis)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="ckpt-g2",
        help="SLURM partition (default: ckpt-g2)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="Number of GPUs per job (default: 1)",
    )
    parser.add_argument(
        "--mem",
        type=int,
        default=128,
        help="Memory in GB (default: 64)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=16,
        help="CPUs per task (default: 8)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="1-00:00:00",
        help="Time limit (default: 1-00:00:00)",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="jarvis_pred",
        help="Base job name (default: jarvis_pred)",
    )
    parser.add_argument(
        "--bouts_root",
        type=str,
        default=None,
        help="If set, switch to bouts mode: scan this directory for "
        "Predictions_3D_* folders, read each info.yaml's recording_path "
        "to map it to a recording under --data_dir, and submit one JARVIS "
        "job per match with --bouts_csv pointed at that folder's "
        "<dataset>_bouts_unified_summary.csv.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="courtship",
        help="Dataset name used to locate <dataset>_bouts_unified_summary.csv "
        "inside each Predictions_3D_* folder under --bouts_root "
        "(default: courtship)",
    )
    parser.add_argument(
        "--stage_suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the stage root directory name. "
        "Without this flag, the stage root is <bouts_root>_bouts (default, "
        "unchanged). With e.g. --stage_suffix $(date +%%m%%d%%Y), it becomes "
        "<bouts_root>_bouts_<suffix>, letting you isolate a re-run from an "
        "existing staged directory. The suffix is a literal string — shell "
        "command substitution happens in your shell, not in Python.",
    )
    parser.add_argument(
        "--predict_script",
        type=str,
        default="jarvis_batch_multi_animal.py",
        help="Python script invoked inside each SLURM job. Path is relative "
             "to the JARVIS-HybridNet root (or absolute). For Phase-4 Option-B "
             "multi-animal 3D with SAM3 bout masks + bout sharding across GPU "
             "pairs, use `tools/predict3D_multianimal_shard.py`.",
    )
    parser.add_argument(
        "--save_clips",
        action="store_true",
        help="Pass --save-clips through to the predict script so per-bout "
             "annotated video clips are written alongside the 3D predictions. "
             "Only meaningful with --predict_script tools/predict3D_multianimal_shard.py.",
    )
    parser.add_argument(
        "--triangulate",
        action="store_true",
        help="Pass --triangulate through: bypass HybridNet 3D fusion and "
             "DLT-triangulate the 2D keypoints (workaround for the v2vNet "
             "courtship collapse). Shard pipeline only.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print jobs that would be submitted without actually submitting",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    # Build the list of (recording_folder, bouts_csv) pairs to submit.
    # In bouts mode, we drive from --bouts_root and map back to recordings
    # via each Predictions_3D_*/info.yaml's recording_path field.
    # In default mode, we walk --data_dir for recording folders (legacy).
    if args.bouts_root:
        print(f"\nBouts mode: scanning {args.bouts_root}")
        mapping = _build_recording_to_bouts_map(
            args.bouts_root, data_dir, args.dataset
        )
        if not mapping:
            print("No Predictions_3D_* folders mapped to recordings; nothing to submit.")
            return
        # Stage root: sibling of --bouts_root with "_bouts" suffix. If
        # --stage_suffix is given, append "_<suffix>" so re-runs can be
        # isolated from existing staged directories (e.g.
        # --stage_suffix $(date +%m%d%Y)).
        bouts_root_p = Path(args.bouts_root).resolve()
        stage_name = f"{bouts_root_p.name}_bouts"
        if args.stage_suffix:
            stage_name = f"{stage_name}_{args.stage_suffix}"
        stage_root = bouts_root_p.parent / stage_name
        submissions = [
            (str(rec), str(csv), str(pred), pred.name) for rec, csv, pred in mapping
        ]
        print(f"\nMapped {len(submissions)} recording(s) from bouts_root:")
        print(f"Stage root (downstream pipeline base-dir): {stage_root}")
        for rec, csv, _src_path, src_name in submissions:
            print(f"  {os.path.basename(rec)}  <-  {src_name}  ({os.path.basename(csv)})")
    else:
        # Legacy full-recording mode.
        has_videos = any(
            os.path.splitext(f)[1].lower() in video_extensions
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
        )
        if has_videos:
            recording_folders = [data_dir]
        else:
            recording_folders = sorted(
                [
                    os.path.join(data_dir, d)
                    for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
            )
        valid_recordings = []
        for folder in recording_folders:
            has_vid = any(
                os.path.splitext(f)[1].lower() in video_extensions
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            )
            if has_vid:
                valid_recordings.append(folder)
            else:
                print(f"Skipping (no videos): {folder}")
        if not valid_recordings:
            print("No recording folders with video files found.")
            sys.exit(1)
        print(f"\nFound {len(valid_recordings)} recording(s) to process:")
        for r in valid_recordings:
            print(f"  {os.path.basename(r)}")
        submissions = [(r, None, None, None) for r in valid_recordings]
        stage_root = None
    print()

    # Ensure OutFiles directory exists
    os.makedirs("OutFiles", exist_ok=True)

    # Submit one job per (recording, optional bouts_csv) pair
    submitted_jobs = []
    for recording_folder, bouts_csv, src_pred_dir, src_name in submissions:
        # Find calibration folder
        calib_folder = os.path.join(recording_folder, "calibration")
        if not os.path.isdir(calib_folder):
            print(f"WARNING: No calibration/ folder in {recording_folder}, skipping")
            continue

        recording_name = os.path.basename(str(recording_folder).rstrip("/"))
        if args.dry_run:
            if bouts_csv:
                print(f"[DRY RUN] Would submit: {recording_name}  bouts={bouts_csv}  (from {src_name})")
            else:
                print(f"[DRY RUN] Would submit: {recording_name}")
            continue

        job_id = submit_predict(
            video_folder=recording_folder,
            calib_folder=calib_folder,
            conda_env_name=args.conda_env_name,
            project=args.project,
            partition=args.partition,
            job_name=args.job_name,
            num_gpus=args.num_gpus,
            mem=args.mem,
            cpus=args.cpus,
            bouts_csv=bouts_csv,
            stage_dir=(
                str(stage_root / f"Predictions_3D_$SLURM_JOB_ID")
                if (stage_root and src_pred_dir)
                else None
            ),
            src_pred_dir=src_pred_dir,
            dataset=args.dataset if src_pred_dir else None,
            time_limit=args.time,
            predict_script_name=args.predict_script,
            save_clips=args.save_clips,
            triangulate=args.triangulate,
        )
        submitted_jobs.append((os.path.basename(recording_folder), job_id))

    # Summary
    if submitted_jobs:
        print(f"\nSubmitted {len(submitted_jobs)} job(s):")
        for name, jid in submitted_jobs:
            print(f"  {jid} -> {name}")
        print(f"\nMonitor with: squeue -u $USER")
        print(f"Cancel all:   squeue -u $USER -h | awk '{{print $1}}' | xargs scancel")


if __name__ == "__main__":
    main()



'''
squeue -u $USER -h -o "%i %j" | awk '/jarvis_pred/ {print $1}' | xargs -r scancel

Inference: 

python ./slurm_predict.py --data_dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 --dry-run

python slurm_predict.py --data_dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 --bouts_root /gscratch/portia/eabe/data/Johnson_lab/courtship/04092026 --stage_suffix $(date +%m%d%Y)


Training: 
CUDA_VISIBLE_DEVICES=0 /home/eabe/miniconda3/envs/jarvis/bin/jarvis-local train all --num_epochs_center 100 --num_epochs_keypoint 200 --num_epochs_hybridnet 100  fly50_V7 



python slurm_predict.py --data_dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0 --bouts_root /gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_04152026_bouts --stage_suffix $(date +%m%d%Y) --project red_data_unified --predict_script tools/predict3D_multianimal_shard.py --num_gpus 4 --mem 128 --cpus 16 --time 1-00:00:00 --dataset courtship
python slurm_predict.py --data_dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0 --bouts_root /gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_04152026_bouts --stage_suffix $(date +%m%d%Y) --project red_data_unified --predict_script tools/predict3D_multianimal_shard.py --num_gpus 4 --mem 128 --cpus 16 --time 1-00:00:00 --dataset courtship

python tools/predict3D_multianimal.py --project red_data_unified --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04 --bouts-csv courtship_bouts_unified_summary.csv --out /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/Predictions_3D_34659165   --bouts 29 --save-masks --save-clips 
'''