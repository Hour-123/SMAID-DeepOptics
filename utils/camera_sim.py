from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_bayer_masks(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    mask_r = (((yy % 2) == 0) & ((xx % 2) == 0)).float()[None, None, ...]
    mask_g = ((((yy % 2) == 0) & ((xx % 2) == 1)) | (((yy % 2) == 1) & ((xx % 2) == 0))).float()[None, None, ...]
    mask_b = (((yy % 2) == 1) & ((xx % 2) == 1)).float()[None, None, ...]
    return mask_r, mask_g, mask_b


def psf_to_otf(psf: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    height, width = output_size
    pad_h = max(height - psf.shape[-2], 0)
    pad_w = max(width - psf.shape[-1], 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = F.pad(psf, (pad_left, pad_right, pad_top, pad_bottom))
    padded = torch.fft.ifftshift(padded, dim=(-2, -1))
    return torch.fft.fft2(padded)


def inverse_filter(blurred: torch.Tensor, estimate: torch.Tensor, psf: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    otf = psf_to_otf(psf, output_size=blurred.shape[-2:])
    img_fft = torch.fft.fft2(blurred)
    estimate_fft = torch.fft.fft2(estimate)
    numerator = img_fft * torch.conj(otf) + gamma * estimate_fft
    denominator = torch.abs(otf) ** 2 + gamma
    result = torch.fft.ifft2(numerator / denominator).real
    return torch.clamp(result, min=1e-5)


def _fft_convolve_same(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel = torch.flip(kernel, dims=(-2, -1))
    return F.conv2d(image, kernel, padding=kernel.shape[-1] // 2)


def simulate_smaid_camera(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    optics,
    noise_std: float = 0.0,
    auto_exposure: bool = True,
    fixed_gain: float = 5.0,
    psf_bank: torch.Tensor | None = None,
    psf_mode: str = "default",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a depth-dependent PSF bank and return a Bayer RAW observation."""
    batch, channels, height, width = rgb.shape
    device = rgb.device
    effective_psf = psf_bank
    if effective_psf is None:
        if hasattr(optics, "psf_for_mode"):
            effective_psf = optics.psf_for_mode(psf_mode)
        else:
            effective_psf = optics.effective_psf()
    lower_idx, upper_idx, lower_weight, upper_weight = optics.depth_to_neighbor_weights(depth)

    sensor_rgb = torch.zeros(batch, channels, height, width, device=device, dtype=rgb.dtype)
    for depth_idx in range(optics.num_depth_layers):
        layer_weight = (
            (lower_idx == depth_idx).to(rgb.dtype) * lower_weight
            + (upper_idx == depth_idx).to(rgb.dtype) * upper_weight
        )
        if layer_weight.sum() == 0:
            continue
        psf_depth = effective_psf[depth_idx]
        for ch in range(channels):
            image_slice = rgb[:, ch : ch + 1] * layer_weight
            kernel = psf_depth[ch : ch + 1].unsqueeze(1)
            sensor_rgb[:, ch : ch + 1] += _fft_convolve_same(image_slice, kernel)

    if hasattr(optics, "effective_qe"):
        qe = optics.effective_qe().to(device=device, dtype=rgb.dtype)
        sensor_rgb_flat = sensor_rgb.permute(0, 2, 3, 1)
        sensor_rgb = torch.matmul(sensor_rgb_flat, qe.t()).permute(0, 3, 1, 2)

    mask_r, mask_g, mask_b = build_bayer_masks(height, width, device)
    raw = sensor_rgb[:, 0:1] * mask_r + sensor_rgb[:, 1:2] * mask_g + sensor_rgb[:, 2:3] * mask_b

    if auto_exposure:
        raw_max = raw.amax()
        gain = 0.95 / raw_max if raw_max > 1e-6 else 1.0
        raw = raw * gain
    else:
        raw = raw * fixed_gain

    if noise_std > 0:
        raw = raw + noise_std * torch.randn_like(raw)

    raw = torch.clamp(raw, 0.0, 1.0)
    depth_clipped = torch.clamp(depth, min=optics.depth_min, max=optics.depth_max)
    return raw, sensor_rgb, effective_psf, depth_clipped
