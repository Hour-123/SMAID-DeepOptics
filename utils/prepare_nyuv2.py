from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm

try:
    import h5py
except ImportError:
    h5py = None


@dataclass
class SplitStats:
    split: str
    images_seen: int = 0
    patches_saved: int = 0
    patches_skipped_low_valid: int = 0
    patches_skipped_low_texture: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NYUv2 RGB-D data into train/val .npz patches for SMAID-DeepOptics."
    )
    parser.add_argument("--raw_root", default="data/raw/nyuv2", help="Directory containing NYUv2 raw files.")
    parser.add_argument(
        "--labeled_mat",
        default=None,
        help="Path to nyu_depth_v2_labeled.mat. Defaults to <raw_root>/nyu_depth_v2_labeled.mat.",
    )
    parser.add_argument(
        "--splits_mat",
        default=None,
        help="Path to splits.mat. Defaults to <raw_root>/splits.mat if it exists.",
    )
    parser.add_argument(
        "--output_root",
        default="data/processed/nyuv2_near_range_patches",
        help="Output directory for processed train/val .npz patches.",
    )
    parser.add_argument(
        "--rgb_key",
        default="images",
        help="MAT key containing RGB images. Usually 'images' for nyu_depth_v2_labeled.mat.",
    )
    parser.add_argument(
        "--depth_key",
        default="depths",
        choices=["depths", "rawDepths"],
        help="Depth source key from the NYUv2 MAT file.",
    )
    parser.add_argument("--patch_size", type=int, default=256, help="Square patch size.")
    parser.add_argument("--stride", type=int, default=128, help="Sliding-window stride.")
    parser.add_argument("--depth_min", type=float, default=0.3, help="Minimum in-range depth in meters.")
    parser.add_argument("--depth_max", type=float, default=1.5, help="Maximum in-range depth in meters.")
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.2,
        help="Minimum fraction of in-range valid depth pixels required to save a patch.",
    )
    parser.add_argument(
        "--min_rgb_std",
        type=float,
        default=0.02,
        help="Minimum normalized RGB standard deviation required to keep a patch.",
    )
    parser.add_argument(
        "--border_crop",
        type=int,
        default=0,
        help="Optional symmetric border crop applied before patch extraction.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio used only when splits.mat is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for fallback random splitting.")
    parser.add_argument(
        "--limit_train_images",
        type=int,
        default=None,
        help="Optional cap on the number of train images to process.",
    )
    parser.add_argument(
        "--limit_val_images",
        type=int,
        default=None,
        help="Optional cap on the number of val images to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing .npz files under output_root before writing new ones.",
    )
    return parser.parse_args()


def default_labeled_mat(raw_root: Path) -> Path:
    return raw_root / "nyu_depth_v2_labeled.mat"


def default_splits_mat(raw_root: Path) -> Path:
    return raw_root / "splits.mat"


def ensure_clean_output(output_root: Path, overwrite: bool) -> tuple[Path, Path]:
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    output_root.mkdir(parents=True, exist_ok=True)

    existing_npz = list(train_dir.glob("*.npz")) + list(val_dir.glob("*.npz"))
    if existing_npz and not overwrite:
        raise RuntimeError(
            f"Found existing .npz files under {output_root}. Re-run with --overwrite to replace them."
        )

    if overwrite:
        for split_dir in (train_dir, val_dir):
            if split_dir.exists():
                shutil.rmtree(split_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir


def infer_sample_axis(shape: tuple[int, ...], sample_name: str) -> int:
    if not shape:
        raise RuntimeError(f"Cannot infer sample axis for empty shape: {sample_name}")
    return int(np.argmax(shape))


def extract_sample(array: Any, index: int, sample_axis: int) -> np.ndarray:
    slices = [slice(None)] * array.ndim
    slices[sample_axis] = index
    return np.asarray(array[tuple(slices)])


def to_hwc_rgb(rgb_sample: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_sample)
    rgb = np.squeeze(rgb)
    if rgb.ndim != 3:
        raise RuntimeError(f"Expected RGB sample with 3 dims, got shape {rgb.shape}")
    if rgb.shape[-1] == 3:
        return rgb
    if rgb.shape[0] == 3:
        return np.transpose(rgb, (1, 2, 0))
    raise RuntimeError(f"Could not identify RGB channel axis from shape {rgb.shape}")


def to_hw_depth(depth_sample: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_sample)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise RuntimeError(f"Expected depth sample with 2 dims, got shape {depth.shape}")
    return depth


def align_modalities(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] == depth.shape:
        return rgb, depth

    rgb_t = np.transpose(rgb, (1, 0, 2))
    if rgb_t.shape[:2] == depth.shape:
        return rgb_t, depth

    depth_t = depth.T
    if rgb.shape[:2] == depth_t.shape:
        return rgb, depth_t
    if rgb_t.shape[:2] == depth_t.shape:
        return rgb_t, depth_t

    raise RuntimeError(f"RGB/depth shapes do not align: rgb={rgb.shape}, depth={depth.shape}")


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def clean_depth(depth: np.ndarray, depth_min: float, depth_max: float) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0) & (depth >= depth_min) & (depth <= depth_max)
    depth_clean = np.nan_to_num(depth, nan=depth_min, posinf=depth_max, neginf=depth_min)
    depth_clean = np.clip(depth_clean, depth_min, depth_max).astype(np.float32)
    return depth_clean, valid.astype(np.float32)


def crop_border(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, border_crop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if border_crop <= 0:
        return rgb, depth, mask
    h, w = depth.shape
    if border_crop * 2 >= h or border_crop * 2 >= w:
        raise RuntimeError(f"border_crop={border_crop} is too large for image size {(h, w)}")
    return (
        rgb[border_crop : h - border_crop, border_crop : w - border_crop],
        depth[border_crop : h - border_crop, border_crop : w - border_crop],
        mask[border_crop : h - border_crop, border_crop : w - border_crop],
    )


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def rgb_texture_std(rgb_patch: np.ndarray) -> float:
    rgb_norm = rgb_patch.astype(np.float32) / 255.0
    return float(rgb_norm.std())


class NYUv2MatReader:
    def __init__(self, labeled_mat_path: Path, rgb_key: str, depth_key: str):
        self.labeled_mat_path = labeled_mat_path
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self._h5_file = None
        self._mat_data: dict[str, Any] | None = None

        try:
            if h5py is None:
                raise ImportError("h5py is not installed")
            self._h5_file = h5py.File(labeled_mat_path, "r")
            self._rgb_array = self._h5_file[rgb_key]
            self._depth_array = self._h5_file[depth_key]
        except (OSError, ImportError, KeyError):
            self._h5_file = None
            try:
                self._mat_data = loadmat(labeled_mat_path)
            except NotImplementedError as exc:
                raise RuntimeError(
                    "This NYUv2 MAT file appears to require HDF5/v7.3 support. "
                    "Install h5py or use a non-v7.3 MAT export."
                ) from exc
            self._rgb_array = self._mat_data[rgb_key]
            self._depth_array = self._mat_data[depth_key]

        self._rgb_sample_axis = infer_sample_axis(tuple(self._rgb_array.shape), rgb_key)
        self._depth_sample_axis = infer_sample_axis(tuple(self._depth_array.shape), depth_key)
        self.num_samples = int(self._rgb_array.shape[self._rgb_sample_axis])
        depth_samples = int(self._depth_array.shape[self._depth_sample_axis])
        if self.num_samples != depth_samples:
            raise RuntimeError(
                f"RGB/depth sample count mismatch: {self.num_samples} vs {depth_samples}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def get_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        rgb = to_hwc_rgb(extract_sample(self._rgb_array, index, self._rgb_sample_axis))
        depth = to_hw_depth(extract_sample(self._depth_array, index, self._depth_sample_axis))
        rgb, depth = align_modalities(rgb, depth)
        return normalize_rgb(rgb), depth.astype(np.float32)

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()


def load_split_indices(num_samples: int, splits_mat_path: Path | None, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if splits_mat_path is not None and splits_mat_path.exists():
        split_data = loadmat(splits_mat_path)
        train_key = "trainNdxs" if "trainNdxs" in split_data else None
        val_key = "valNdxs" if "valNdxs" in split_data else ("testNdxs" if "testNdxs" in split_data else None)
        if train_key and val_key:
            train_indices = np.asarray(split_data[train_key]).reshape(-1).astype(np.int64) - 1
            val_indices = np.asarray(split_data[val_key]).reshape(-1).astype(np.int64) - 1
            return train_indices, val_indices

    if not 0.0 < val_ratio < 1.0:
        raise RuntimeError("val_ratio must be within (0, 1) when splits.mat is unavailable.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    val_count = max(1, int(round(num_samples * val_ratio)))
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


def maybe_limit(indices: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    return indices[:limit]


def save_patch(
    out_path: Path,
    rgb_patch: np.ndarray,
    depth_patch: np.ndarray,
    mask_patch: np.ndarray,
) -> None:
    np.savez_compressed(
        out_path,
        rgb=rgb_patch.astype(np.uint8),
        depth=depth_patch.astype(np.float32),
        mask=mask_patch.astype(np.float32),
    )


def process_split(
    split_name: str,
    indices: np.ndarray,
    out_dir: Path,
    reader: NYUv2MatReader,
    args: argparse.Namespace,
) -> SplitStats:
    stats = SplitStats(split=split_name)
    patch_counter = 0

    for sample_index in tqdm(indices, desc=f"Preparing {split_name}", unit="image"):
        rgb, depth = reader.get_pair(int(sample_index))
        depth_clipped, mask = clean_depth(depth, depth_min=args.depth_min, depth_max=args.depth_max)
        rgb, depth_clipped, mask = crop_border(rgb, depth_clipped, mask, args.border_crop)

        positions_y = sliding_positions(depth_clipped.shape[0], args.patch_size, args.stride)
        positions_x = sliding_positions(depth_clipped.shape[1], args.patch_size, args.stride)
        stats.images_seen += 1

        for top in positions_y:
            for left in positions_x:
                rgb_patch = rgb[top : top + args.patch_size, left : left + args.patch_size]
                depth_patch = depth_clipped[top : top + args.patch_size, left : left + args.patch_size]
                mask_patch = mask[top : top + args.patch_size, left : left + args.patch_size]

                valid_ratio = float(mask_patch.mean())
                if valid_ratio < args.min_valid_ratio:
                    stats.patches_skipped_low_valid += 1
                    continue

                if rgb_texture_std(rgb_patch) < args.min_rgb_std:
                    stats.patches_skipped_low_texture += 1
                    continue

                out_path = out_dir / f"{split_name}_{patch_counter:07d}.npz"
                save_patch(out_path, rgb_patch, depth_patch, mask_patch)
                patch_counter += 1
                stats.patches_saved += 1

    return stats


def write_summary(
    output_root: Path,
    args: argparse.Namespace,
    reader: NYUv2MatReader,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_stats: SplitStats,
    val_stats: SplitStats,
) -> None:
    summary = {
        "source": {
            "labeled_mat": str((Path(args.labeled_mat) if args.labeled_mat else default_labeled_mat(Path(args.raw_root))).resolve()),
            "splits_mat": str((Path(args.splits_mat).resolve()) if args.splits_mat else default_splits_mat(Path(args.raw_root)).resolve()),
            "rgb_key": args.rgb_key,
            "depth_key": args.depth_key,
            "num_samples_total": len(reader),
            "num_train_images": int(len(train_indices)),
            "num_val_images": int(len(val_indices)),
        },
        "preprocess": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "min_valid_ratio": args.min_valid_ratio,
            "min_rgb_std": args.min_rgb_std,
            "border_crop": args.border_crop,
        },
        "stats": {
            "train": asdict(train_stats),
            "val": asdict(val_stats),
        },
    }
    with (output_root / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    labeled_mat_path = Path(args.labeled_mat) if args.labeled_mat else default_labeled_mat(raw_root)
    splits_mat_path = Path(args.splits_mat) if args.splits_mat else default_splits_mat(raw_root)
    output_root = Path(args.output_root)

    if not labeled_mat_path.exists():
        raise FileNotFoundError(
            f"Could not find labeled NYUv2 MAT file at {labeled_mat_path}. "
            "Set --labeled_mat explicitly if your file lives elsewhere."
        )

    train_dir, val_dir = ensure_clean_output(output_root, overwrite=args.overwrite)
    reader = NYUv2MatReader(labeled_mat_path=labeled_mat_path, rgb_key=args.rgb_key, depth_key=args.depth_key)
    try:
        train_indices, val_indices = load_split_indices(
            num_samples=len(reader),
            splits_mat_path=splits_mat_path if splits_mat_path.exists() else None,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_indices = maybe_limit(train_indices, args.limit_train_images)
        val_indices = maybe_limit(val_indices, args.limit_val_images)

        train_stats = process_split("train", train_indices, train_dir, reader, args)
        val_stats = process_split("val", val_indices, val_dir, reader, args)
        write_summary(output_root, args, reader, train_indices, val_indices, train_stats, val_stats)

        print("Finished preparing NYUv2 patches.")
        print(f"Train patches: {train_stats.patches_saved}")
        print(f"Val patches:   {val_stats.patches_saved}")
        print(f"Output root:   {output_root}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
