from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class IOException(Exception):
    pass


class PatchDataset(Dataset):
    """Dataset for processed RGB/depth/mask patches stored as .npz files."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)
        rgb = data["rgb"].astype(np.float32) / 255.0
        depth = data["depth"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

        rgb = np.transpose(rgb, (2, 0, 1))
        depth = depth[None, ...]
        mask = mask[None, ...]
        return torch.from_numpy(rgb), torch.from_numpy(depth), torch.from_numpy(mask)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_ldr(img: np.ndarray | torch.Tensor, file_path: str) -> None:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    ensure_dir(os.path.dirname(file_path))
    plt.imsave(file_path, img)


def save_psf_bank(psf_bank: torch.Tensor, out_dir: str) -> None:
    ensure_dir(out_dir)
    psf_np = psf_bank.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "psf_bank.npy"), psf_np)
    num_depths, num_channels = psf_np.shape[:2]
    representative_depths = sorted({0, num_depths // 2, num_depths - 1})

    fig, axes = plt.subplots(len(representative_depths), num_channels, figsize=(4 * num_channels, 4 * len(representative_depths)))
    axes = np.atleast_2d(axes)
    channel_names = ["R", "G", "B"]

    for row_idx, depth_idx in enumerate(representative_depths):
        for channel_idx in range(num_channels):
            psf = psf_np[depth_idx, channel_idx]
            psf = psf / (psf.max() + 1e-8)
            ax = axes[row_idx, channel_idx]
            ax.imshow(psf, cmap="inferno")
            ax.set_title(f"depth {depth_idx:02d}, ch {channel_names[channel_idx] if channel_idx < len(channel_names) else channel_idx}")
            ax.axis("off")

            plt.figure(figsize=(4, 4))
            plt.imshow(psf, cmap="inferno")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"psf_depth_{depth_idx:02d}_ch_{channel_idx:02d}.png"))
            plt.close()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psf_representative_overview.png"))
    plt.close(fig)


def save_full_psf_visualizations(psf_bank: torch.Tensor, out_dir: str) -> None:
    ensure_dir(out_dir)
    psf_np = psf_bank.detach().cpu().numpy()
    num_depths, num_channels = psf_np.shape[:2]
    channel_names = ["R", "G", "B"]

    for mode in ("linear", "log"):
        fig, axes = plt.subplots(num_depths, num_channels, figsize=(4 * num_channels, 3.4 * num_depths))
        axes = np.atleast_2d(axes)
        for depth_idx in range(num_depths):
            for channel_idx in range(num_channels):
                psf = psf_np[depth_idx, channel_idx]
                if mode == "linear":
                    vis = psf / (psf.max() + 1e-8)
                else:
                    vis = np.log10(np.clip(psf / (psf.max() + 1e-8), 1e-8, None))
                ax = axes[depth_idx, channel_idx]
                im = ax.imshow(vis, cmap="inferno")
                ax.set_title(
                    f"depth {depth_idx:02d}, ch {channel_names[channel_idx] if channel_idx < len(channel_names) else channel_idx}"
                )
                ax.axis("off")
                if channel_idx == num_channels - 1:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"psf_all_depths_{mode}.png"))
        plt.close(fig)


def save_phase_mask(phase_mask: torch.Tensor, out_dir: str, stem: str = "phase_mask") -> None:
    ensure_dir(out_dir)
    phase_np = phase_mask.detach().cpu().numpy()
    np.save(os.path.join(out_dir, f"{stem}.npy"), phase_np)
    for idx in range(phase_np.shape[0]):
        height_map = phase_np[idx]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(height_map, cmap="viridis")
        ax.set_title("Height Map (meters)")
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("meters")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_{idx:02d}.png"))
        plt.close(fig)


def save_sample_visualization(
    raw: torch.Tensor,
    rgb_gt: torch.Tensor,
    rgb_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    out_dir: str,
    prefix: str,
    max_samples: int = 4,
) -> None:
    ensure_dir(out_dir)
    batch = min(max_samples, raw.shape[0])
    raw_np = raw.detach().cpu().numpy()
    rgb_gt_np = rgb_gt.detach().cpu().numpy()
    rgb_pred_np = rgb_pred.detach().cpu().numpy()
    depth_gt_np = depth_gt.detach().cpu().numpy()
    depth_pred_np = depth_pred.detach().cpu().numpy()

    for idx in range(batch):
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        axes[0, 0].imshow(raw_np[idx, 0], cmap="gray")
        axes[0, 0].set_title("RAW")
        axes[0, 1].imshow(np.transpose(np.clip(rgb_gt_np[idx], 0, 1), (1, 2, 0)))
        axes[0, 1].set_title("RGB GT")
        axes[0, 2].imshow(np.transpose(np.clip(rgb_pred_np[idx], 0, 1), (1, 2, 0)))
        axes[0, 2].set_title("RGB Pred")
        im1 = axes[1, 0].imshow(depth_gt_np[idx, 0], cmap="viridis")
        axes[1, 0].set_title("Depth GT")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        im2 = axes[1, 1].imshow(depth_pred_np[idx, 0], cmap="viridis")
        axes[1, 1].set_title("Depth Pred")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        error = np.abs(depth_pred_np[idx, 0] - depth_gt_np[idx, 0])
        im3 = axes[1, 2].imshow(error, cmap="magma")
        axes[1, 2].set_title("Depth Error")
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_sample_{idx:02d}.png"))
        plt.close(fig)


def latest_checkpoint(output_dir: str) -> str:
    ckpt_path = Path(output_dir) / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise IOException(f"Checkpoint not found: {ckpt_path}")
    return str(ckpt_path)
