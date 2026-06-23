# Running JARVIS-HybridNet in the `3d_tracking` conda env

The legacy `jarvis` env (Py3.10 / numpy 1.26) still works. As of 2026-06-20 the
`3d_tracking` env (Py3.12 / numpy 2.2, `/gscratch/portia/eabe/miniconda3/envs/3d_tracking`)
also runs JARVIS-HybridNet end-to-end (train + predict, both verified with output
identical to the jarvis env on the same weights).

## What was done

1. **Editable-install JARVIS + SAM3 (no-deps, to protect torch 2.11+cu130):**
   ```bash
   pip install -e third_party/JARVIS-HybridNet --no-deps      # creates the jarvis-local entry point
   pip install -e /gscratch/portia/eabe/Research/Github/sam3 --no-deps
   ```
2. **Missing deps** (torch pinned via a constraints file so nothing downgrades it):
   `timm streamlit albumentations streamlit_option_menu inquirer yacs imgaug seaborn`
   and for SAM3: `iopath>=0.1.10 ftfy regex` (hydra-core/omegaconf/einops already present).
3. **numpy-2 compat for imgaug** — `jarvis/utils/numpy2_compat.py` restores the
   numpy aliases imgaug needs (`np.sctypes`, `np.product`, `np.float_`, …) that
   were removed in numpy 2.0. Imported at the top of `jarvis/dataset/dataset2D.py`
   and `dataset3D.py` before imgaug. No-op on numpy 1.x, so the jarvis env is
   unaffected.
4. **streamlit ≥1.12 compat** — `import streamlit.cli` (removed upstream) was made
   lazy in `jarvis/ui/jarvis.py` so the train/predict CLI works; only the GUI
   `launch` command still needs it.

## Runtime requirement (important)

Two env vars are needed after activating:
```bash
micromamba activate 3d_tracking
# 1. py3.12 wheels (cv2, PIL, torchvision) need a newer libstdc++ than /lib64
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
# 2. SAM3 predict triggers an nvrtc JIT that dlopens libnvrtc-builtins.so.13.0;
#    expose torch's bundled cu13 libs (only needed for the SAM3 predict path).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
```
The first is enough for *training*; SAM3 *prediction* needs both. The SLURM
training scripts (`scripts/train_hybridnet_v3.slurm`,
`3d_tracking_dataset/scripts/slurm_train.py`) set `LD_PRELOAD`. A 3d_tracking
SAM3-predict launcher must also set `LD_LIBRARY_PATH` as above.

## Verified (2026-06-20)

- `jarvis-local --help` and the train/predict subcommands.
- Dataset2D (KeypointDetect + CenterDetect, train+val) and Dataset3D
  (train+val) — imgaug augmentation runs on real data under numpy 2.2.
- Predict path (build_models + KeypointDetect + HybridNet 3D fusion) — produces
  output identical to the jarvis env on the same weights.
- `SAM3VideoTracker` (sam3.1) builds and loads.

## Optional / not installed

- `flash_attn` is **not** installed (and never was, in either env). SAM3 runs on
  PyTorch's built-in SDPA flash kernels (`flash_sdp_enabled() == True`). On this
  L40S (Ada sm_89) a flash-attn-3 build is Hopper-first and low-ROI; there are no
  prebuilt wheels for torch 2.11+cu130, so it would be a fragile source build.
- **Run SAM3 with `--no-sam3-compile` in this env.** Benchmark (Session0 bout 1,
  100 frames, 7 cams, L40S): compile OFF = clean, ~13.5–14.2 s/cam propagation
  (SAM3+ID 218 s incl. model load). compile ON = torch.compile/inductor floods
  `triton … OutOfMemoryError: out of resource … Required 131072 > Hardware limit
  101376` (Ada shared-memory limit), falls back, and the warmup never finished
  one camera in >3 min. Not viable here; flash-attn would not fix this (it's an
  inductor/triton issue, and SDPA-flash is already active).
