import argparse
import os
import subprocess
import sys


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
    # predict_script = os.path.join(jarvis_root, "jarvis_batch_frame_range.py")
    predict_script = os.path.join(jarvis_root, "jarvis_batch_multi_animal.py")

    # Derive a short name from the recording folder
    recording_name = os.path.basename(video_folder.rstrip("/"))

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
conda activate {conda_env_name}
unset LD_LIBRARY_PATH
echo $SLURMD_NODENAME
echo "Processing recording: {video_folder}"
python -u {predict_script} \\
    --project {project} \\
    --video_folder {video_folder} \\
    --calib_folder {calib_folder} \\
    --num_animals 2 \\
    --no_sam3_mask \\
    --gpus 0 1 2 3 \\
    --output_name $SLURM_JOB_ID
"""
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
        default="merge_courtship_V3",
        help="JARVIS project name (default: merge_courtship_V3)",
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
        "--dry_run",
        action="store_true",
        help="Print jobs that would be submitted without actually submitting",
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    # Determine if data_dir is a session directory (contains recording folders)
    # or a single recording folder (contains video files)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    has_videos = any(
        os.path.splitext(f)[1].lower() in video_extensions
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    )

    if has_videos:
        # data_dir is a single recording folder
        recording_folders = [data_dir]
    else:
        # data_dir is a session directory; find recording subfolders
        recording_folders = sorted(
            [
                os.path.join(data_dir, d)
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )

    # Filter to folders that actually contain video files
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
    print()

    # Ensure OutFiles directory exists
    os.makedirs("OutFiles", exist_ok=True)

    # Submit one job per recording
    submitted_jobs = []
    for recording_folder in valid_recordings:
        # Find calibration folder
        calib_folder = os.path.join(recording_folder, "calibration")
        if not os.path.isdir(calib_folder):
            print(f"WARNING: No calibration/ folder in {recording_folder}, skipping")
            continue

        if args.dry_run:
            recording_name = os.path.basename(recording_folder.rstrip("/"))
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
            time_limit=args.time,
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
