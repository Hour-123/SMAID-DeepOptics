from __future__ import annotations

import csv
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_history(history: list[dict[str, Any]], out_dir: str) -> None:
    ensure_dir(out_dir)
    write_json(os.path.join(out_dir, "history.json"), history)
    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(os.path.join(out_dir, "history.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def compute_psf_stats(psf_bank: torch.Tensor) -> dict[str, Any]:
    psf = psf_bank.detach().cpu().float()
    flat = psf.view(psf.shape[0], psf.shape[1], -1)
    peak = flat.max(dim=-1).values
    entropy = -(flat * torch.log(flat.clamp_min(1e-12))).sum(dim=-1)

    center = psf.shape[-1] // 2
    y0 = max(center - 1, 0)
    y1 = min(center + 2, psf.shape[-2])
    x0 = max(center - 1, 0)
    x1 = min(center + 2, psf.shape[-1])
    center_mass = psf[..., y0:y1, x0:x1].sum(dim=(-2, -1))

    yy, xx = torch.meshgrid(
        torch.arange(psf.shape[-2], dtype=torch.float32),
        torch.arange(psf.shape[-1], dtype=torch.float32),
        indexing="ij",
    )
    rr2 = (yy - center) ** 2 + (xx - center) ** 2
    second_moment = (psf * rr2).sum(dim=(-2, -1))

    center21_mask = torch.ones((psf.shape[-2], psf.shape[-1]), dtype=torch.bool)
    center21_mask[max(center - 10, 0) : min(center + 11, psf.shape[-2]), max(center - 10, 0) : min(center + 11, psf.shape[-1])] = False
    outside_center21_max = flat[..., center21_mask.view(-1)].max(dim=-1).values

    border = max(1, psf.shape[-1] // 16)
    edge_mask = torch.zeros((psf.shape[-2], psf.shape[-1]), dtype=torch.bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    edge_energy = psf[..., edge_mask].sum(dim=-1)

    per_depth = []
    for depth_idx in range(psf.shape[0]):
        channels = []
        for channel_idx in range(psf.shape[1]):
            channels.append(
                {
                    "channel_idx": channel_idx,
                    "peak": float(peak[depth_idx, channel_idx]),
                    "center_3x3_mass": float(center_mass[depth_idx, channel_idx]),
                    "entropy": float(entropy[depth_idx, channel_idx]),
                    "second_moment": float(second_moment[depth_idx, channel_idx]),
                    "outside_center21_max": float(outside_center21_max[depth_idx, channel_idx]),
                    "edge_energy": float(edge_energy[depth_idx, channel_idx]),
                }
            )
        per_depth.append({"depth_idx": depth_idx, "channels": channels})

    return {
        "peak_mean": float(peak.mean()),
        "peak_min": float(peak.min()),
        "peak_max": float(peak.max()),
        "center_3x3_mass_mean": float(center_mass.mean()),
        "entropy_mean": float(entropy.mean()),
        "entropy_min": float(entropy.min()),
        "entropy_max": float(entropy.max()),
        "second_moment_mean": float(second_moment.mean()),
        "outside_center21_max_mean": float(outside_center21_max.mean()),
        "edge_energy_mean": float(edge_energy.mean()),
        "per_depth": per_depth,
    }


def compute_qe_stats(qe_matrix: torch.Tensor | None) -> dict[str, Any]:
    if qe_matrix is None:
        return {}

    qe = qe_matrix.detach().cpu().float()
    diag = torch.diag(qe)
    off_diag_mask = ~torch.eye(qe.shape[0], dtype=torch.bool)
    off_diag = qe[off_diag_mask]
    diag_ratio = diag.sum() / qe.sum().clamp_min(1e-12)
    return {
        "matrix": qe.tolist(),
        "diag_mean": float(diag.mean()),
        "off_diag_mean": float(off_diag.mean()),
        "diag_ratio": float(diag_ratio),
    }


def plot_history_curves(history: list[dict[str, Any]], out_dir: str) -> None:
    if not history:
        return

    ensure_dir(out_dir)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_L_total"] for row in history], label="train total")
    plt.plot(epochs, [row["val_L_total"] for row in history], label="val total")
    plt.plot(epochs, [row["train_L_rgb"] for row in history], label="train rgb")
    plt.plot(epochs, [row["val_L_rgb"] for row in history], label="val rgb")
    plt.plot(epochs, [row["train_L_depth"] for row in history], label="train depth")
    plt.plot(epochs, [row["val_L_depth"] for row in history], label="val depth")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_it_per_s"] for row in history], label="train it/s")
    plt.plot(epochs, [row["val_it_per_s"] for row in history], label="val it/s")
    plt.xlabel("epoch")
    plt.ylabel("iterations / second")
    plt.title("Throughput")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speed_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["psf_peak_mean"] for row in history], label="psf peak mean")
    plt.plot(epochs, [row["psf_center_3x3_mass_mean"] for row in history], label="psf center 3x3 mean")
    plt.plot(epochs, [row["psf_entropy_mean"] for row in history], label="psf entropy mean")
    if "psf_edge_energy_mean" in history[0]:
        plt.plot(epochs, [row["psf_edge_energy_mean"] for row in history], label="psf edge energy mean")
    plt.xlabel("epoch")
    plt.ylabel("metric value")
    plt.title("PSF Statistics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psf_curve.png"))
    plt.close()

    if "qe_diag_ratio" in history[0]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["qe_diag_ratio"] for row in history], label="qe diag ratio")
        plt.plot(epochs, [row["qe_diag_mean"] for row in history], label="qe diag mean")
        plt.plot(epochs, [row["qe_off_diag_mean"] for row in history], label="qe off-diag mean")
        plt.xlabel("epoch")
        plt.ylabel("metric value")
        plt.title("QE Statistics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "qe_curve.png"))
        plt.close()


def save_qe_heatmap(qe_matrix: torch.Tensor, out_dir: str, filename: str = "qe_matrix_final.png") -> None:
    ensure_dir(out_dir)
    qe = qe_matrix.detach().cpu().float().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(qe, cmap="viridis", vmin=0.0, vmax=max(float(qe.max()), 1e-6))
    plt.xticks(range(qe.shape[1]), ["R", "G", "B"][: qe.shape[1]])
    plt.yticks(range(qe.shape[0]), ["R", "G", "B"][: qe.shape[0]])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Final QE Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def summarize_epoch(
    *,
    epoch: int,
    train_stats: dict[str, float],
    val_stats: dict[str, float],
    train_steps: int,
    val_steps: int,
    train_seconds: float,
    val_seconds: float,
    psf_stats: dict[str, Any],
    qe_stats: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "train_L_total": train_stats["L_total"],
        "train_L_rgb": train_stats["L_rgb"],
        "train_L_depth": train_stats["L_depth"],
        "val_L_total": val_stats["L_total"],
        "val_L_rgb": val_stats["L_rgb"],
        "val_L_depth": val_stats["L_depth"],
        "train_steps": train_steps,
        "val_steps": val_steps,
        "train_seconds": train_seconds,
        "val_seconds": val_seconds,
        "train_it_per_s": train_steps / max(train_seconds, 1e-8),
        "val_it_per_s": val_steps / max(val_seconds, 1e-8),
        "psf_peak_mean": psf_stats["peak_mean"],
        "psf_peak_min": psf_stats["peak_min"],
        "psf_peak_max": psf_stats["peak_max"],
        "psf_center_3x3_mass_mean": psf_stats["center_3x3_mass_mean"],
        "psf_entropy_mean": psf_stats["entropy_mean"],
        "psf_second_moment_mean": psf_stats["second_moment_mean"],
        "psf_outside_center21_max_mean": psf_stats["outside_center21_max_mean"],
        "psf_edge_energy_mean": psf_stats["edge_energy_mean"],
    }
    if qe_stats:
        row["qe_diag_mean"] = qe_stats["diag_mean"]
        row["qe_off_diag_mean"] = qe_stats["off_diag_mean"]
        row["qe_diag_ratio"] = qe_stats["diag_ratio"]
    return row
