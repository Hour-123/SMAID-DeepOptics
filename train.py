import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import network, optics
from utils import camera_sim, img_io, log_utils


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch training script inspired by DeepOpticsHDR.")
    parser.add_argument("--data_dir", default="data/processed/nyuv2_near_range_patches", help="Processed data root.")
    parser.add_argument("--output_dir", default="outputs", help="Experiment output root.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Train batches per epoch; defaults to full train loader.")
    parser.add_argument("--val_steps", type=int, default=None, help="Validation batches per epoch; defaults to full val loader.")
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--psf_learning_rate", type=float, default=1.0e-2)
    parser.add_argument("--lambda_depth", type=float, default=20.0)
    parser.add_argument("--lambda_rgb", type=float, default=2.0)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--optics_type", default="direct_psf", choices=["direct_psf", "phase_mask"])
    parser.add_argument("--psf_size", type=int, default=127)
    parser.add_argument("--psf_init", default="delta", choices=["delta", "gaussian", "random"])
    parser.add_argument("--num_depth_layers", type=int, default=8)
    parser.add_argument("--depth_min", type=float, default=0.3)
    parser.add_argument("--depth_max", type=float, default=1.5)
    parser.add_argument("--focus_distance", type=float, default=0.9)
    parser.add_argument("--wavelengths", type=float, nargs=3, default=(635e-9, 530e-9, 450e-9))
    parser.add_argument("--refractive_indices", type=float, nargs=3, default=(1.4295, 1.4349, 1.4421))
    parser.add_argument("--pupil_radius", type=float, default=1.0e-3)
    parser.add_argument("--wave_resolution", type=int, default=255)
    parser.add_argument("--phase_mask_size", type=float, default=3.0e-3)
    parser.add_argument("--sensor_pixel_size", type=float, default=None)
    parser.add_argument("--focal_length", type=float, default=35e-3)
    parser.add_argument("--sensor_distance", type=float, default=None)
    parser.add_argument("--no_thin_lens", action="store_true", help="Disable the explicit thin-lens focusing phase in phase-mask optics.")
    parser.add_argument("--height_map_max", type=float, default=1.55e-6)
    parser.add_argument("--height_map_noise_std", type=float, default=0.0)
    parser.add_argument("--height_quantization_res", type=float, default=21.16e-9)
    parser.add_argument("--laplace_reg", type=float, default=0.0)
    parser.add_argument("--psf_edge_reg", type=float, default=1e-3)
    parser.add_argument("--quantization_reg", type=float, default=0.0)
    parser.add_argument(
        "--quantization_reg_anneal",
        action="store_true",
        help="Linearly ramp quantization regularization from a smaller starting value to --quantization_reg.",
    )
    parser.add_argument(
        "--quantization_reg_anneal_start_scale",
        type=float,
        default=0.1,
        help="Starting scale relative to --quantization_reg when annealing is enabled.",
    )
    parser.add_argument(
        "--quantization_reg_anneal_start_frac",
        type=float,
        default=0.0,
        help="Normalized epoch fraction where quantization annealing begins.",
    )
    parser.add_argument(
        "--quantization_reg_anneal_end_frac",
        type=float,
        default=1.0,
        help="Normalized epoch fraction where quantization annealing reaches --quantization_reg.",
    )
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--auto_exposure", action="store_true", default=True)
    parser.add_argument("--fixed_gain", type=float, default=5.0)
    parser.add_argument("--fix_qe", action="store_true", help="Keep the QE matrix fixed at its initialization.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="directpsf_run")
    parser.add_argument("--num_visualization_samples", type=int, default=4, help="Random validation samples to save after the final epoch.")
    parser.add_argument("--save_full_psf_visualizations", action="store_true", help="Save full 16/24-panel PSF overview images in addition to representative slices.")
    return parser.parse_args()


def get_device(device_flag: str) -> torch.device:
    if device_flag == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_dir(root: str, run_name: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, f"{run_name}_{timestamp}")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "im"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "psf"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "phase"), exist_ok=True)
    return out_dir


def loss_fn(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: torch.Tensor,
    lambda_depth: float,
    lambda_rgb: float,
    optics_reg: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = mask > 0.5
    rgb_diff = (rgb_pred - rgb_gt) * mask
    rgb_loss = rgb_diff.abs()[valid.expand_as(rgb_diff)].mean() if valid.any() else rgb_pred.new_tensor(0.0)
    depth_diff = depth_pred[valid] - depth_gt[valid] if valid.any() else depth_pred.new_zeros((1,))
    depth_loss = (depth_diff**2).mean() if valid.any() else depth_pred.new_tensor(0.0)
    optics_reg_loss = optics_reg if optics_reg is not None else rgb_pred.new_tensor(0.0)
    total = lambda_rgb * rgb_loss + lambda_depth * depth_loss + optics_reg_loss
    return total, {
        "L_total": float(total.item()),
        "L_rgb": float(rgb_loss.item()),
        "L_depth": float(depth_loss.item()),
        "L_optics_reg": float(optics_reg_loss.item()),
    }


def quantization_reg_for_epoch(args, epoch: int) -> float:
    target = float(args.quantization_reg)
    if not getattr(args, "quantization_reg_anneal", False) or target <= 0:
        return target

    start_scale = max(0.0, float(args.quantization_reg_anneal_start_scale))
    start_frac = float(args.quantization_reg_anneal_start_frac)
    end_frac = float(args.quantization_reg_anneal_end_frac)
    if not 0.0 <= start_frac <= 1.0:
        raise ValueError("--quantization_reg_anneal_start_frac must be within [0, 1].")
    if not 0.0 <= end_frac <= 1.0:
        raise ValueError("--quantization_reg_anneal_end_frac must be within [0, 1].")
    if end_frac < start_frac:
        raise ValueError("--quantization_reg_anneal_end_frac must be >= start_frac.")

    epoch_frac = 1.0 if args.epochs <= 1 else epoch / (args.epochs - 1)
    start_value = target * start_scale
    if epoch_frac <= start_frac:
        return start_value
    if epoch_frac >= end_frac:
        return target
    ramp = (epoch_frac - start_frac) / max(end_frac - start_frac, 1e-8)
    return start_value + ramp * (target - start_value)


def export_modes_for_model(optics_model) -> tuple[str, ...]:
    if hasattr(optics_model, "quantized_height_map") and hasattr(optics_model, "psf_for_mode"):
        return ("continuous", "quantized")
    return ("default",)


def export_subdir(base_dir: str, mode: str) -> str:
    return base_dir if mode == "default" else os.path.join(base_dir, mode)


def save_random_validation_samples(
    dataset: img_io.PatchDataset,
    optics_model,
    recon_model,
    device: torch.device,
    args,
    out_dir: str,
    prefix: str,
    sample_indices: list[int] | None = None,
    psf_mode: str = "default",
) -> list[int]:
    sample_count = min(args.num_visualization_samples, len(dataset))
    if sample_count <= 0:
        return []

    if sample_indices is None:
        sample_indices = random.SystemRandom().sample(range(len(dataset)), k=sample_count)
    rgb_batch = []
    depth_batch = []
    for idx in sample_indices:
        rgb_gt, depth_gt, _ = dataset[idx]
        rgb_batch.append(rgb_gt)
        depth_batch.append(depth_gt)

    rgb_gt = torch.stack(rgb_batch, dim=0).to(device)
    depth_gt = torch.stack(depth_batch, dim=0).to(device)

    raw, _, _, depth_clipped = camera_sim.simulate_smaid_camera(
        rgb_gt,
        depth_gt,
        optics_model,
        noise_std=args.noise_std,
        auto_exposure=args.auto_exposure,
        fixed_gain=args.fixed_gain,
        psf_mode=psf_mode,
    )
    rgb_pred, depth_norm = recon_model(raw)
    depth_pred = args.depth_min + depth_norm * (args.depth_max - args.depth_min)
    img_io.save_sample_visualization(
        raw=raw,
        rgb_gt=rgb_gt,
        rgb_pred=rgb_pred,
        depth_gt=depth_clipped,
        depth_pred=depth_pred,
        out_dir=out_dir,
        prefix=prefix,
        max_samples=sample_count,
    )
    return sample_indices


def build_optics_model(args, device: torch.device):
    if args.optics_type == "phase_mask":
        model = optics.PhaseMaskOptics(
            psf_size=args.psf_size,
            num_depth_layers=args.num_depth_layers,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            focus_distance=args.focus_distance,
            psf_init=args.psf_init,
            wavelengths=tuple(args.wavelengths),
            refractive_indices=tuple(args.refractive_indices),
            pupil_radius=args.pupil_radius,
            wave_resolution=args.wave_resolution,
            phase_mask_size=args.phase_mask_size,
            sensor_pixel_size=args.sensor_pixel_size,
            focal_length=args.focal_length,
            sensor_distance=args.sensor_distance,
            use_thin_lens=not args.no_thin_lens,
            height_map_max=args.height_map_max,
            height_map_noise_std=args.height_map_noise_std,
            height_quantization_res=args.height_quantization_res,
            laplace_reg=args.laplace_reg,
            psf_edge_reg=args.psf_edge_reg,
            quantization_reg=args.quantization_reg,
        )
    else:
        model = optics.DirectPSFOptics(
            psf_size=args.psf_size,
            num_depth_layers=args.num_depth_layers,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            focus_distance=args.focus_distance,
            psf_init=args.psf_init,
        )
    return model.to(device)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(vars(args))
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_ds = img_io.PatchDataset(train_dir)
    val_ds = img_io.PatchDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_steps = len(train_loader) if args.steps_per_epoch is None else min(len(train_loader), args.steps_per_epoch)
    val_steps = len(val_loader) if args.val_steps is None else min(len(val_loader), args.val_steps)
    print(f"Train steps per epoch: {train_steps} / {len(train_loader)}")
    print(f"Val steps per epoch: {val_steps} / {len(val_loader)}")

    optical_model = build_optics_model(args, device)
    if args.fix_qe and hasattr(optical_model, "W_qe"):
        optical_model.W_qe.requires_grad_(False)
    recon_model = network.my_medium_model(in_channels=1, base_channels=args.base_channels).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": optical_model.parameters(), "lr": args.psf_learning_rate, "weight_decay": 0.0},
            {"params": recon_model.parameters(), "lr": args.learning_rate, "weight_decay": 1.0e-4},
        ]
    )

    out_dir = make_output_dir(args.output_dir, args.run_name)
    logs_dir = os.path.join(out_dir, "logs")
    print(f"Saving outputs to: {out_dir}")
    if hasattr(optical_model, "sampling_diagnostics"):
        log_utils.write_json(os.path.join(logs_dir, "sampling_diagnostics.json"), optical_model.sampling_diagnostics())
    best_val = float("inf")
    best_epoch = -1
    last_train_stats = None
    last_val_stats = None
    history: list[dict[str, float]] = []
    psf_stats_history: list[dict[str, object]] = []
    height_stats_history: list[dict[str, object]] = []
    quantization_reg_history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        recon_model.train()
        optical_model.train()
        current_quantization_reg = quantization_reg_for_epoch(args, epoch)
        if hasattr(optical_model, "quantization_reg"):
            optical_model.quantization_reg = current_quantization_reg
        running = {"L_total": 0.0, "L_rgb": 0.0, "L_depth": 0.0, "L_optics_reg": 0.0}
        step_count = 0
        train_start = time.perf_counter()
        train_progress = tqdm(
            enumerate(train_loader),
            total=train_steps,
            desc=f"Train Epoch {epoch}",
        )
        for step, (rgb_gt, depth_gt, mask) in train_progress:
            if step >= train_steps:
                break
            rgb_gt = rgb_gt.to(device)
            depth_gt = depth_gt.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            raw, _, _, depth_clipped = camera_sim.simulate_smaid_camera(
                rgb_gt,
                depth_gt,
                optical_model,
                noise_std=args.noise_std,
                auto_exposure=args.auto_exposure,
                fixed_gain=args.fixed_gain,
            )
            rgb_pred, depth_norm = recon_model(raw)
            depth_pred = args.depth_min + depth_norm * (args.depth_max - args.depth_min)
            optics_reg = optical_model.regularization_loss()
            loss, stats = loss_fn(
                rgb_pred,
                rgb_gt,
                depth_pred,
                depth_clipped,
                mask,
                args.lambda_depth,
                args.lambda_rgb,
                optics_reg=optics_reg,
            )
            loss.backward()
            optimizer.step()

            for key in running:
                running[key] += stats[key]
            step_count += 1

        for key in running:
            running[key] /= max(1, step_count)
        train_seconds = time.perf_counter() - train_start
        last_train_stats = running
        print(
            f"[Train] Epoch {epoch} stats: {running} | "
            f"qreg={current_quantization_reg:.6g} | {step_count / max(train_seconds, 1e-8):.2f} it/s"
        )

        recon_model.eval()
        optical_model.eval()
        val_running = {"L_total": 0.0, "L_rgb": 0.0, "L_depth": 0.0, "L_optics_reg": 0.0}
        val_count = 0
        val_start = time.perf_counter()
        with torch.no_grad():
            val_progress = tqdm(
                enumerate(val_loader),
                total=val_steps,
                desc=f"Val Epoch {epoch}",
            )
            for step, (rgb_gt, depth_gt, mask) in val_progress:
                if step >= val_steps:
                    break
                rgb_gt = rgb_gt.to(device)
                depth_gt = depth_gt.to(device)
                mask = mask.to(device)

                raw, _, _, depth_clipped = camera_sim.simulate_smaid_camera(
                    rgb_gt,
                    depth_gt,
                    optical_model,
                    noise_std=args.noise_std,
                    auto_exposure=args.auto_exposure,
                    fixed_gain=args.fixed_gain,
                )
                rgb_pred, depth_norm = recon_model(raw)
                depth_pred = args.depth_min + depth_norm * (args.depth_max - args.depth_min)
                optics_reg = optical_model.regularization_loss()
                _, stats = loss_fn(
                    rgb_pred,
                    rgb_gt,
                    depth_pred,
                    depth_clipped,
                    mask,
                    args.lambda_depth,
                    args.lambda_rgb,
                    optics_reg=optics_reg,
                )
                for key in val_running:
                    val_running[key] += stats[key]
                val_count += 1

        for key in val_running:
            val_running[key] /= max(1, val_count)
        val_seconds = time.perf_counter() - val_start
        last_val_stats = val_running
        print(f"[Val] Epoch {epoch} stats: {val_running} | {val_count / max(val_seconds, 1e-8):.2f} it/s")

        with torch.no_grad():
            psf_stats = log_utils.compute_psf_stats(optical_model.effective_psf())
            height_stats = optical_model.height_map_stats() if hasattr(optical_model, "height_map_stats") else {}
        psf_stats_history.append({"epoch": epoch, **psf_stats})
        if height_stats:
            height_stats_history.append({"epoch": epoch, **height_stats})
        quantization_reg_history.append({"epoch": epoch, "quantization_reg": current_quantization_reg})

        history_row = log_utils.summarize_epoch(
            epoch=epoch,
            train_stats=running,
            val_stats=val_running,
            train_steps=step_count,
            val_steps=val_count,
            train_seconds=train_seconds,
            val_seconds=val_seconds,
            psf_stats=psf_stats,
            qe_stats={},
        )
        history_row["quantization_reg"] = current_quantization_reg
        history.append(history_row)
        log_utils.write_history(history, logs_dir)
        log_utils.write_json(os.path.join(logs_dir, "psf_stats_history.json"), psf_stats_history)
        if height_stats_history:
            log_utils.write_json(os.path.join(logs_dir, "height_stats_history.json"), height_stats_history)
        log_utils.write_json(os.path.join(logs_dir, "quantization_reg_history.json"), quantization_reg_history)
        log_utils.plot_history_curves(history, logs_dir)

        checkpoint = {
            "epoch": epoch,
            "args": vars(args),
            "optics": optical_model.state_dict(),
            "network": recon_model.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "train_stats": running,
            "val_stats": val_running,
        }
        torch.save(checkpoint, os.path.join(out_dir, "checkpoints", "last.pt"))
        if val_running["L_total"] < best_val:
            best_val = val_running["L_total"]
            best_epoch = epoch
            checkpoint["best_val"] = best_val
            checkpoint["best_epoch"] = best_epoch
            torch.save(checkpoint, os.path.join(out_dir, "checkpoints", "best.pt"))

    best_ckpt_path = os.path.join(out_dir, "checkpoints", "best.pt")
    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    optical_model.load_state_dict(best_checkpoint["optics"], strict=False)
    recon_model.load_state_dict(best_checkpoint["network"])
    optical_model.eval()
    recon_model.eval()

    export_modes = export_modes_for_model(optical_model)
    sample_indices = None
    with torch.no_grad():
        exported_psfs = {}
        for mode in export_modes:
            sample_indices = save_random_validation_samples(
                dataset=val_ds,
                optics_model=optical_model,
                recon_model=recon_model,
                device=device,
                args=args,
                out_dir=export_subdir(os.path.join(out_dir, "im"), mode),
                prefix=f"best_epoch_{best_checkpoint['epoch']:03d}",
                sample_indices=sample_indices,
                psf_mode=mode,
            )
            exported_psfs[mode] = optical_model.psf_for_mode(mode)
    for mode, effective_psf in exported_psfs.items():
        psf_dir = export_subdir(os.path.join(out_dir, "psf"), mode)
        img_io.save_psf_bank(effective_psf, psf_dir)
        if args.save_full_psf_visualizations:
            img_io.save_full_psf_visualizations(effective_psf, psf_dir)
    if args.optics_type == "phase_mask" and hasattr(optical_model, "height_map"):
        phase_dir = os.path.join(out_dir, "phase")
        img_io.save_phase_mask(optical_model.height_map().unsqueeze(0), phase_dir, stem="height_map_continuous")
        if hasattr(optical_model, "quantized_height_map"):
            img_io.save_phase_mask(optical_model.quantized_height_map().unsqueeze(0), phase_dir, stem="height_map_quantized")
    summary = {
        "output_dir": out_dir,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "last_train_stats": last_train_stats,
        "last_val_stats": last_val_stats,
        "export_modes": list(export_modes),
        "quantization_reg_history": quantization_reg_history,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved checkpoints, images, PSF visualizations, and summary to: {out_dir}")


if __name__ == "__main__":
    main()
