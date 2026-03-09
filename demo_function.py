import argparse
import os

import torch

from src import network, optics
from utils import camera_sim, img_io


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch demo script inspired by DeepOpticsHDR.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--sample", type=int, default=0, help="Validation sample index.")
    parser.add_argument("--data_dir", default="data/processed/nyuv2_near_range_patches/val", help="Validation set path.")
    parser.add_argument("--output_dir", default="Reconstructions", help="Output directory for demo images.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def get_device(device_flag: str) -> torch.device:
    if device_flag == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optics_model(train_args: dict, device: torch.device):
    optics_type = train_args.get("optics_type", "direct_psf")
    if optics_type == "phase_mask":
        model = optics.PhaseMaskOptics(
            psf_size=train_args["psf_size"],
            num_depth_layers=train_args["num_depth_layers"],
            depth_min=train_args["depth_min"],
            depth_max=train_args["depth_max"],
            focus_distance=train_args["focus_distance"],
            psf_init=train_args["psf_init"],
            wavelengths=tuple(train_args.get("wavelengths", (635e-9, 530e-9, 450e-9))),
            refractive_indices=tuple(train_args.get("refractive_indices", (1.4295, 1.4349, 1.4421))),
            pupil_radius=train_args.get("pupil_radius", 1.0e-3),
            wave_resolution=train_args.get("wave_resolution", 255),
            phase_mask_size=train_args.get("phase_mask_size", 3.0e-3),
            sensor_pixel_size=train_args.get("sensor_pixel_size"),
            focal_length=train_args.get("focal_length", 35e-3),
            sensor_distance=train_args.get("sensor_distance", 35e-3),
            use_thin_lens=not train_args.get("no_thin_lens", False),
            height_map_max=train_args.get("height_map_max", 1.55e-6),
            height_map_noise_std=train_args.get("height_map_noise_std", 0.0),
            height_quantization_res=train_args.get("height_quantization_res", 21.16e-9),
            laplace_reg=train_args.get("laplace_reg", 0.0),
            psf_edge_reg=train_args.get("psf_edge_reg", 0.0),
            quantization_reg=train_args.get("quantization_reg", 0.0),
        )
    else:
        model = optics.DirectPSFOptics(
            psf_size=train_args["psf_size"],
            num_depth_layers=train_args["num_depth_layers"],
            depth_min=train_args["depth_min"],
            depth_max=train_args["depth_max"],
            focus_distance=train_args["focus_distance"],
            psf_init=train_args["psf_init"],
        )
    return model.to(device)


def export_modes_for_model(optics_model) -> tuple[str, ...]:
    if hasattr(optics_model, "quantized_height_map") and hasattr(optics_model, "psf_for_mode"):
        return ("continuous", "quantized")
    return ("default",)


def export_subdir(base_dir: str, mode: str) -> str:
    return base_dir if mode == "default" else os.path.join(base_dir, mode)


def main():
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint["args"]

    optical_model = build_optics_model(train_args, device)
    recon_model = network.my_medium_model(in_channels=1, base_channels=train_args["base_channels"]).to(device)
    optical_model.load_state_dict(checkpoint["optics"], strict=False)
    recon_model.load_state_dict(checkpoint["network"])
    optical_model.eval()
    recon_model.eval()

    dataset = img_io.PatchDataset(args.data_dir)
    rgb_gt, depth_gt, _ = dataset[args.sample]
    rgb_gt = rgb_gt.unsqueeze(0).to(device)
    depth_gt = depth_gt.unsqueeze(0).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    export_modes = export_modes_for_model(optical_model)
    with torch.no_grad():
        exported_psfs = {}
        for mode in export_modes:
            raw, _, effective_psf, depth_clipped = camera_sim.simulate_smaid_camera(
                rgb_gt,
                depth_gt,
                optical_model,
                noise_std=train_args["noise_std"],
                auto_exposure=train_args["auto_exposure"],
                fixed_gain=train_args["fixed_gain"],
                psf_mode=mode,
            )
            rgb_pred, depth_norm = recon_model(raw)
            depth_pred = train_args["depth_min"] + depth_norm * (train_args["depth_max"] - train_args["depth_min"])
            img_io.save_sample_visualization(
                raw=raw,
                rgb_gt=rgb_gt,
                rgb_pred=rgb_pred,
                depth_gt=depth_clipped,
                depth_pred=depth_pred,
                out_dir=export_subdir(args.output_dir, mode),
                prefix="demo",
                max_samples=1,
            )
            exported_psfs[mode] = effective_psf
    for mode, effective_psf in exported_psfs.items():
        img_io.save_psf_bank(effective_psf, export_subdir(os.path.join(args.output_dir, "psf"), mode))
    if train_args.get("optics_type") == "phase_mask" and hasattr(optical_model, "height_map"):
        phase_dir = os.path.join(args.output_dir, "phase")
        img_io.save_phase_mask(optical_model.height_map().unsqueeze(0), phase_dir, stem="height_map_continuous")
        if hasattr(optical_model, "quantized_height_map"):
            img_io.save_phase_mask(optical_model.quantized_height_map().unsqueeze(0), phase_dir, stem="height_map_quantized")


if __name__ == "__main__":
    main()
