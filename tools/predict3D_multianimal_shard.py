"""Shard wrapper around `predict3D_multianimal.py` for multi-GPU SLURM jobs.

Given N GPUs (N even), spawns N/2 subprocesses of the single-bout pipeline,
each with a disjoint pair of physical GPUs (JARVIS on ch 0, SAM3 on ch 1
within each subprocess's CUDA_VISIBLE_DEVICES view). Splits the recording's
bouts evenly across shards.

CLI is intentionally SLURM-wrapper-compatible: the same flags that
`slurm_predict.py` already passes (`--video_folder`, `--calib_folder`,
`--num_animals`, `--num_gpus`, `--output_name`, `--bouts_csv`, `--project`)
are accepted here.
"""

import argparse
import csv
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_SCRIPT = os.path.join(ROOT, 'tools/predict3D_multianimal.py')


def read_bout_indices(csv_path, session_tag):
    """Return list of bout_idx values filtered to rows whose fly_id matches
    session_tag (None → no filter). Mirrors parse_bouts in the main script."""
    out = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if session_tag and r.get('fly_id') != session_tag:
                continue
            out.append(int(r['bout_idx']))
    return out


def shard_list(items, n):
    """Round-robin shard — keeps long bouts roughly balanced across workers."""
    shards = [[] for _ in range(n)]
    for i, x in enumerate(items):
        shards[i % n].append(x)
    return shards


def main():
    ap = argparse.ArgumentParser()
    # SLURM-wrapper compatibility args:
    ap.add_argument('--project', default='red_data_unified')
    ap.add_argument('--video_folder', required=True,
                    help='Recording directory (CamXXX.mp4 + calibration/).')
    ap.add_argument('--calib_folder', default=None,
                    help='Ignored — calibration comes from project config.')
    ap.add_argument('--num_animals', type=int, default=2)
    ap.add_argument('--num_gpus', type=int, default=4,
                    help='Total GPUs for this job. Each shard uses 2 '
                         '(JARVIS + SAM3); num_shards = num_gpus // 2.')
    ap.add_argument('--output_name', required=True,
                    help='Suffix for the per-recording output directory: '
                         '<video_folder>/Predictions_3D_<output_name>/')
    ap.add_argument('--bouts_csv', required=True)
    # Passthroughs:
    ap.add_argument('--center-weights', default=None)
    ap.add_argument('--kp-weights', default=None)
    ap.add_argument('--hybridnet-weights', default=None)
    ap.add_argument('--sam3-text', default='insect')
    ap.add_argument('--sam3-version', default='sam3',
                    choices=['sam3', 'sam3.1'])
    ap.add_argument('--sam3-compile', dest='sam3_compile',
                    action='store_true', default=True)
    ap.add_argument('--no-sam3-compile', dest='sam3_compile',
                    action='store_false')
    ap.add_argument('--sam3-checkpoint', default=None)
    ap.add_argument('--save-masks', action='store_true')
    ap.add_argument('--reuse-masks', action='store_true')
    ap.add_argument('--save-overlays-every', type=int, default=0)
    ap.add_argument('--save-clips', action='store_true')
    args = ap.parse_args()

    if args.num_gpus % 2 != 0 or args.num_gpus < 2:
        print(f'ERROR: --num_gpus must be even and ≥ 2 (got {args.num_gpus})',
              file=sys.stderr)
        sys.exit(2)
    num_shards = args.num_gpus // 2

    session_dir = os.path.abspath(args.video_folder)
    session_tag = '/'.join(session_dir.rstrip('/').split('/')[-2:])
    bout_ids = read_bout_indices(args.bouts_csv, session_tag)
    if not bout_ids:
        print(f'No bouts in {args.bouts_csv} for session {session_tag}')
        sys.exit(0)

    shards = shard_list(bout_ids, num_shards)
    out_dir = os.path.join(
        session_dir, f'Predictions_3D_{args.output_name}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[shard] session={session_tag} out={out_dir}', flush=True)
    print(f'[shard] {len(bout_ids)} bouts across {num_shards} shard(s):',
          flush=True)
    for i, s in enumerate(shards):
        print(f'  shard {i} (gpu {2*i}:{2*i+1}): {len(s)} bouts '
              f'→ {s[:10]}{"..." if len(s) > 10 else ""}', flush=True)

    # Inherit env, override CUDA_VISIBLE_DEVICES per subprocess so each one
    # sees exactly two GPUs (JARVIS=0, SAM3=1 from the subprocess's view).
    procs = []
    for i, s in enumerate(shards):
        if not s:
            continue
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = f'{2 * i},{2 * i + 1}'

        cmd = [
            sys.executable, SINGLE_SCRIPT,
            '--project', args.project,
            '--session', session_dir,
            '--bouts-csv', args.bouts_csv,
            '--out', out_dir,
            '--num-animals', str(args.num_animals),
            '--sam3-gpu', '1',
            '--sam3-text', args.sam3_text,
            '--bouts', ','.join(str(b) for b in s),
        ]
        for flag, val in [('--center-weights', args.center_weights),
                          ('--kp-weights', args.kp_weights),
                          ('--hybridnet-weights', args.hybridnet_weights)]:
            if val is not None:
                cmd += [flag, val]
        cmd += ['--sam3-version', args.sam3_version]
        if not args.sam3_compile:
            cmd += ['--no-sam3-compile']
        if args.sam3_checkpoint is not None:
            cmd += ['--sam3-checkpoint', args.sam3_checkpoint]
        if args.save_masks:
            cmd += ['--save-masks']
        if args.reuse_masks:
            cmd += ['--reuse-masks']
        if args.save_overlays_every:
            cmd += ['--save-overlays-every', str(args.save_overlays_every)]
        if args.save_clips:
            cmd += ['--save-clips']

        log_path = os.path.join(out_dir, f'shard_{i}.log')
        log_f = open(log_path, 'w')
        print(f'[shard] launch {i} → {log_path}', flush=True)
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((i, p, log_f, log_path))

    # Wait; surface any non-zero exit.
    failures = []
    for i, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        status = 'OK' if rc == 0 else f'FAIL (rc={rc})'
        print(f'[shard] shard {i} done: {status}  log={log_path}', flush=True)
        if rc != 0:
            failures.append(i)

    if failures:
        print(f'[shard] {len(failures)} shard(s) failed: {failures}',
              file=sys.stderr)
        sys.exit(1)
    print('[shard] all shards OK')


if __name__ == '__main__':
    main()
