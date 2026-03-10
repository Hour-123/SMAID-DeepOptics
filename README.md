# SMAID-DeepOptics

PyTorch reimplementation scaffold inspired by `DeepOpticsHDR`, adapted for SMAID-style single-shot monocular all-in-focus imaging and depth sensing.

Current focus:
- `DirectPSF` training path
- `PhaseMask` training path
- depth-dependent PSF bank
- RAW camera simulation from RGB-D patches
- joint RGB + depth reconstruction

## Project Layout

```text
SMAID-DeepOptics/
├── .venv/
├── data/
├── requirements.txt
├── TechnicalSummary.md
├── README.md
├── train.py
├── demo_function.py
├── visualize_psf_bank.py
├── src/
│   ├── network.py
│   └── optics.py
└── utils/
    ├── camera_sim.py
    ├── img_io.py
    ├── log_utils.py
    └── prepare_nyuv2.py
```

## Quick Start

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 2. Verify dependencies

If needed:

```bash
pip install -r requirements.txt
```

### 3. Confirm processed data exists

The training script expects processed near-range patches here:

```text
data/processed/nyuv2_near_range_patches/
├── train/
└── val/
```

Each `.npz` patch should contain:
- `rgb`
- `depth`
- `mask`

## Data Download And Preprocessing

`data/` is intentionally not tracked in git. The current repository contains the training / inference code, but not the dataset itself.

What is already implemented in code:

- `train.py` and `demo_function.py` load processed patches from `data/processed/nyuv2_near_range_patches/...`
- `utils/img_io.py` implements `PatchDataset`, which reads per-sample `.npz` files
- `utils/prepare_nyuv2.py` converts raw NYUv2 data into train/val `.npz` patches
- each `.npz` is expected to contain:
  - `rgb`: RGB image patch, stored as `H x W x 3`
  - `depth`: depth patch, stored as `H x W`
  - `mask`: valid near-range depth mask, stored as `H x W`

What is not implemented in this repo yet:

- no dataset downloader
- no script that fetches NYUv2 or any other RGB-D dataset

Expected raw NYUv2 layout:

```text
data/raw/nyuv2/
├── nyu_depth_v2_labeled.mat
└── splits.mat                       # optional but recommended
```

The preprocessing script is designed around the common NYUv2 MATLAB release:

- RGB key: `images`
- depth key: `depths` by default
- optional depth key: `rawDepths`
- split keys from `splits.mat`: `trainNdxs` and `testNdxs` or `valNdxs`

Recommended workflow if you want to prepare data:

1. Download NYUv2 manually and place the raw files under `data/raw/nyuv2/`.
2. Run the preprocessing script to create train / val patches:

```bash
python utils/prepare_nyuv2.py \
  --raw_root data/raw/nyuv2 \
  --output_root data/processed/nyuv2_near_range_patches \
  --patch_size 256 \
  --stride 128 \
  --depth_min 0.3 \
  --depth_max 1.5 \
  --overwrite
```

3. The script writes patches under:

```text
data/processed/nyuv2_near_range_patches/
├── train/
└── val/
```

4. Point `--data_dir` to that processed root when running training or demo.

What `utils/prepare_nyuv2.py` does:

1. Read aligned RGB and depth pairs from `nyu_depth_v2_labeled.mat`.
2. Use `splits.mat` if present; otherwise fall back to a random train/val split.
3. Clip depth into the configured range, default `[0.3, 1.5]` meters.
4. Build `mask` from finite, positive, in-range depth pixels.
5. Optionally crop borders with `--border_crop`.
6. Tile each frame into overlapping square patches with `--patch_size` and `--stride`.
7. Skip patches with too little valid depth or too little RGB texture.
8. Save each accepted patch as one `.npz` file.
9. Write `prepare_summary.json` under the output root.

Useful preprocessing options:

```text
--depth_key          depths | rawDepths
--patch_size         default 256
--stride             default 128
--depth_min          default 0.3
--depth_max          default 1.5
--min_valid_ratio    default 0.2
--min_rgb_std        default 0.02
--border_crop        default 0
--overwrite          replace existing processed patches
```

Example patch export:

```python
np.savez_compressed(
    out_path,
    rgb=rgb_uint8_hw3,
    depth=depth_float32_hw,
    mask=mask_float32_hw,
)
```

Important status note:

- today, the repo can train and infer from processed `.npz` patches
- today, the repo can preprocess raw NYUv2 `.mat` data into `.npz` patches
- today, the repo still does not download NYUv2 automatically

## Train a Model

Minimal smoke run:

```bash
python train.py --epochs 1 --steps_per_epoch 2 --val_steps 1 --num_workers 0 --run_name smoke_directpsf
```

Longer example:

```bash
python train.py \
  --epochs 5 \
  --psf_size 127 \
  --num_workers 0 \
  --run_name delta_5ep_trend
```

Phase-mask smoke run:

```bash
python train.py \
  --optics_type phase_mask \
  --fix_qe \
  --epochs 1 \
  --steps_per_epoch 1 \
  --val_steps 1 \
  --num_workers 0 \
  --run_name phase_mask_smoke
```

Useful training arguments:

```text
--psf_init           delta | gaussian | random
--optics_type        direct_psf | phase_mask
--psf_size           PSF kernel size, default 127
--num_depth_layers   number of discrete depth bins, default 8
--depth_min          default 0.3
--depth_max          default 1.5
--focus_distance     default 0.9
--wavelengths        default 635e-9 530e-9 450e-9
--refractive_indices default 1.4295 1.4349 1.4421
--pupil_radius       default 1.0e-3
--wave_resolution    default 255
--phase_mask_size    default 3.0e-3
--sensor_distance    default 35e-3
--height_map_max     default 1.55e-6
--height_map_noise_std manufacturing height noise std in meters
--height_quantization_res height quantization resolution in meters
--laplace_reg        default 0.0
--quantization_reg   soft penalty that pulls heights toward quantized levels
--quantization_reg_anneal enable linear annealing for quantization regularization
--quantization_reg_anneal_start_scale starting scale relative to --quantization_reg, default 0.1
--quantization_reg_anneal_start_frac normalized epoch fraction where annealing begins
--quantization_reg_anneal_end_frac normalized epoch fraction where annealing reaches target
--psf_learning_rate  default 1e-2
--learning_rate      default 1e-4
--fix_qe             keep QE matrix fixed at initialization
--num_visualization_samples random val samples saved at final epoch, default 4
--save_full_psf_visualizations save all depth/channel PSF overview images
```

## Where Results Are Saved

Every training run creates a timestamped experiment directory under:

```text
outputs/experiments/<run_name>_<timestamp>/
```

Typical contents:

```text
checkpoints/
  best.pt
  last.pt
im/
  best_epoch_000_sample_00.png              # direct_psf
  ...
  continuous/                               # phase_mask
    best_epoch_000_sample_00.png
  quantized/
    best_epoch_000_sample_00.png
psf/
  psf_bank.npy                              # direct_psf
  psf_depth_00_ch_00.png
  ...
  continuous/                               # phase_mask
    psf_bank.npy
    psf_depth_00_ch_00.png
    ...
  quantized/
    psf_bank.npy
    psf_depth_00_ch_00.png
    ...
phase/
  height_map_continuous.npy                 # phase_mask
  height_map_continuous_00.png
  height_map_quantized.npy
  height_map_quantized_00.png
logs/
  history.json
  history.csv
  loss_curve.png
  speed_curve.png
  psf_curve.png
  psf_stats_history.json
  quantization_reg_history.json
summary.json
```

The training script now prints the exact output directory at startup and shutdown. Final `im/` and `psf/` exports are generated from `best.pt`, not the last epoch checkpoint.

For `phase_mask`, the final export now writes both continuous and quantized comparisons from the same best checkpoint:
- `im/continuous` vs `im/quantized`
- `psf/continuous` vs `psf/quantized`

When quantization annealing is enabled, the per-epoch coefficient is written to:
- `logs/quantization_reg_history.json`
- `summary.json` under `quantization_reg_history`

## Run Inference

Example:

```bash
python demo_function.py \
  --checkpoint outputs/experiments/smoke_directpsf_YYYYMMDD_HHMMSS/checkpoints/last.pt \
  --sample 0 \
  --output_dir outputs/demo_smoke
```

Inference outputs include:
- `demo_sample_00.png` for `direct_psf`
- `continuous/demo_sample_00.png` and `quantized/demo_sample_00.png` for `phase_mask`
- `psf/` visualizations
- `phase/height_map_continuous*` and `phase/height_map_quantized*` for `phase_mask`

## Visualize a Saved PSF Bank

Use the standalone tool to render a saved `psf_bank.npy` into overview images after training:

```bash
python visualize_psf_bank.py \
  --input outputs/experiments/your_run/psf/psf_bank.npy \
  --output_dir outputs/experiments/your_run/psf_full
```

This saves:
- `psf_representative_overview.png`
- `psf_all_depths_linear.png`
- `psf_all_depths_log.png`

The tool also re-saves a copy of `psf_bank.npy` in the target directory so the visualizations and the tensor snapshot stay together.

## Current Camera Model

The current camera simulator in `utils/camera_sim.py` does the following:

1. Maps each depth value to its two nearest depth layers.
2. Uses linear interpolation between the neighboring PSFs.
3. Accumulates a blurred `sensor_rgb`.
4. Applies a `3x3` QE mixing matrix before Bayer sampling.
5. Applies an RGGB Bayer mask to produce single-channel RAW.
6. Applies auto exposure or fixed gain.
7. Optionally adds simple Gaussian noise.

Current simplifications:
- Depth uses only nearest two-layer interpolation rather than a richer continuous PSF model
- QE is currently a compact `3x3` channel-mixing approximation rather than a full spectral response curve
- Evaluation is still centered on training losses; standard image/depth metrics are not yet reported in the training loop

## Current Optical Metadata

The current RGB channel wavelength metadata in `src/optics.py` is:

- R: `630e-9`
- G: `530e-9`
- B: `460e-9`

These values are currently stored as metadata for future physics-based extensions.

## Notes

- Both `DirectPSF` and `PhaseMaskOptics` are implemented training paths.
- `PhaseMaskOptics` now supports continuous and quantized exports for PSF and height-map inspection.
- RAW outputs may look visually blurry because they are blurred, depth-layered, Bayer-sampled measurements rather than display-ready RGB images.
