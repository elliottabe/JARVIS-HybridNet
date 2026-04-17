"""Thin wrapper to launch EfficientTrack training programmatically so we
can set CUDA_VISIBLE_DEVICES and capture logs without relying on the
click-based CLI.

Usage:
    python tools/run_train.py CenterDetect red_data_unified --epochs 100 \
        --pretrain MonkeyHand
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['CenterDetect', 'KeypointDetect'])
    ap.add_argument('project')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--pretrain', default='MonkeyHand',
                    help="'MonkeyHand', 'EcoSet', 'latest', 'None', or a "
                         'weights .pth path')
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()

    import jarvis.train_interface as ti
    weights = args.pretrain
    if weights == 'None':
        weights = None
    ok = ti.train_efficienttrack(args.mode, args.project, args.epochs,
                                 weights, run_name=args.run_name)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
