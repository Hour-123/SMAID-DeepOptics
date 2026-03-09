from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OpticalMetadata:
    depth_min: float
    depth_max: float
    depth_layers: list[float]
    focus_distance: float
    psf_size: int
    wavelengths: list[float]
    qe_matrix: list[list[float]]


class BaseOptics(nn.Module):
    def depth_to_bins(self, depth: torch.Tensor) -> torch.Tensor:
        depth_layers = self.depth_layers.to(depth.device).view(1, self.num_depth_layers, 1, 1)
        return torch.argmin(torch.abs(depth - depth_layers), dim=1)

    def depth_to_neighbor_weights(
        self, depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth_clipped = torch.clamp(depth, min=self.depth_min, max=self.depth_max)
        if self.num_depth_layers <= 1:
            zero_idx = torch.zeros_like(depth_clipped, dtype=torch.long)
            one_weight = torch.ones_like(depth_clipped)
            zero_weight = torch.zeros_like(depth_clipped)
            return zero_idx, zero_idx, one_weight, zero_weight

        layer_pos = (depth_clipped - self.depth_min) / (self.depth_max - self.depth_min)
        layer_pos = layer_pos * (self.num_depth_layers - 1)
        lower_idx = torch.floor(layer_pos).long().clamp(min=0, max=self.num_depth_layers - 1)
        upper_idx = (lower_idx + 1).clamp(max=self.num_depth_layers - 1)

        upper_weight = (layer_pos - lower_idx.to(layer_pos.dtype)).clamp(min=0.0, max=1.0)
        same_bin = upper_idx == lower_idx
        upper_weight = torch.where(same_bin, torch.zeros_like(upper_weight), upper_weight)
        lower_weight = 1.0 - upper_weight
        return lower_idx, upper_idx, lower_weight, upper_weight

    def effective_qe(self) -> torch.Tensor:
        qe = torch.abs(self.W_qe)
        return qe / (qe.sum(dim=1, keepdim=True) + 1e-8)

    def psf_for_mode(self, mode: str = "default") -> torch.Tensor:
        if mode not in {"default", "continuous", "quantized"}:
            raise ValueError(f"Unsupported PSF mode: {mode}")
        return self.effective_psf()


class DirectPSFOptics(BaseOptics):
    """Directly learn a depth-dependent PSF bank, inspired by DeepOpticsHDR DirectPSF."""

    def __init__(
        self,
        psf_size: int = 91,
        num_depth_layers: int = 8,
        num_channels: int = 3,
        wavelengths: tuple[float, float, float] = (630e-9, 530e-9, 460e-9),
        depth_min: float = 0.3,
        depth_max: float = 1.5,
        focus_distance: float = 0.9,
        psf_init: str = "delta",
        tie_psf: bool = False,
    ):
        super().__init__()
        self.psf_size = psf_size
        self.num_depth_layers = num_depth_layers
        self.num_channels = num_channels
        self.wavelengths = tuple(wavelengths)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.focus_distance = focus_distance
        self.tie_psf = tie_psf
        self.register_buffer(
            "depth_layers",
            torch.linspace(depth_min, depth_max, num_depth_layers),
            persistent=False,
        )

        channels_to_learn = 1 if tie_psf else num_channels
        init_kernel = self._initial_kernel(psf_init)
        init_logits = torch.log(torch.clamp(init_kernel, min=1e-8))
        init_logits = init_logits.repeat(num_depth_layers, channels_to_learn, 1, 1)
        self.psf_param = nn.Parameter(init_logits)
        self.W_qe = nn.Parameter(
            torch.tensor(
                [
                    [0.6737, 0.1069, 0.0501],
                    [0.1332, 0.9490, 0.2304],
                    [0.0409, 0.2324, 0.9220],
                ],
                dtype=torch.float32,
            )
        )

    def _initial_kernel(self, mode: str) -> torch.Tensor:
        size = self.psf_size
        center = size // 2
        kernel = torch.full((size, size), 1e-6, dtype=torch.float32)

        if mode == "delta":
            kernel[center, center] = 1.0
        elif mode == "gaussian":
            yy, xx = torch.meshgrid(
                torch.arange(size, dtype=torch.float32),
                torch.arange(size, dtype=torch.float32),
                indexing="ij",
            )
            sigma = max(size / 12.0, 1.0)
            kernel = torch.exp(-((yy - center) ** 2 + (xx - center) ** 2) / (2.0 * sigma**2))
        elif mode == "random":
            kernel = torch.rand((size, size), dtype=torch.float32) + 1e-3
        else:
            raise ValueError(f"Unsupported psf_init: {mode}")

        return kernel / kernel.sum()

    def metadata(self) -> OpticalMetadata:
        return OpticalMetadata(
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            depth_layers=[float(x) for x in self.depth_layers.detach().cpu().tolist()],
            focus_distance=self.focus_distance,
            psf_size=self.psf_size,
            wavelengths=[float(x) for x in self.wavelengths],
            qe_matrix=self.effective_qe().detach().cpu().tolist(),
        )

    def effective_psf(self) -> torch.Tensor:
        flat = self.psf_param.view(self.psf_param.shape[0], self.psf_param.shape[1], -1)
        flat = torch.softmax(flat, dim=-1)
        psf = flat.view_as(self.psf_param)
        if self.tie_psf:
            psf = psf.repeat(1, self.num_channels, 1, 1)
        return psf

    def psf_for_depth(self, depth_idx: int) -> torch.Tensor:
        return self.effective_psf()[depth_idx]

    def psf_slice(self, depth_idx: int, channel_idx: int) -> torch.Tensor:
        return self.psf_for_depth(depth_idx)[channel_idx]

    def regularization_loss(self) -> torch.Tensor:
        return self.psf_param.new_tensor(0.0)


class PhaseMaskOptics(BaseOptics):
    """Learn a shared phase-mask height map and derive a depth-dependent PSF bank."""

    def __init__(
        self,
        psf_size: int = 127,
        num_depth_layers: int = 8,
        num_channels: int = 3,
        wavelengths: tuple[float, float, float] = (635e-9, 530e-9, 450e-9),
        refractive_indices: tuple[float, float, float] = (1.4295, 1.4349, 1.4421),
        depth_min: float = 0.3,
        depth_max: float = 1.5,
        focus_distance: float = 0.9,
        psf_init: str = "delta",
        pupil_radius: float = 1.0e-3,
        wave_resolution: int = 255,
        phase_mask_size: float = 3.0e-3,
        focal_length: float = 35e-3,
        sensor_distance: float | None = None,
        use_thin_lens: bool = True,
        sensor_pixel_size: float | None = None,
        height_map_max: float = 1.55e-6,
        height_map_noise_std: float = 0.0,
        laplace_reg: float = 0.0,
        psf_edge_reg: float = 0.0,
        height_quantization_res: float = 21.16e-9,
        quantization_reg: float = 0.0,
    ):
        super().__init__()
        self.psf_size = psf_size
        self.num_depth_layers = num_depth_layers
        self.num_channels = num_channels
        self.wavelengths = tuple(wavelengths)
        self.refractive_indices = tuple(refractive_indices)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.focus_distance = focus_distance
        self.pupil_radius = pupil_radius
        self.wave_resolution = wave_resolution
        self.phase_mask_size = phase_mask_size
        self.focal_length = focal_length
        self.use_thin_lens = use_thin_lens
        self.sensor_distance = sensor_distance if sensor_distance is not None else self._sensor_distance_for_focus()
        self.wave_feature_size = phase_mask_size / wave_resolution
        # Default to a 1:1 mapping between wave-grid samples and sensor pixels.
        # This keeps the exported PSF focused on the central sensor field of view
        # instead of averaging almost the entire propagated field back into the kernel.
        self.sensor_pixel_size = sensor_pixel_size if sensor_pixel_size is not None else self.wave_feature_size
        self.height_map_max = height_map_max
        self.height_map_noise_std = height_map_noise_std
        self.laplace_reg = laplace_reg
        self.psf_edge_reg = psf_edge_reg
        self.height_quantization_res = height_quantization_res
        self.quantization_reg = quantization_reg
        self.register_buffer(
            "depth_layers",
            torch.linspace(depth_min, depth_max, num_depth_layers),
            persistent=False,
        )
        self.register_buffer(
            "refractive_indices_tensor",
            torch.tensor(refractive_indices, dtype=torch.float32),
            persistent=False,
        )

        half_extent = 0.5 * phase_mask_size
        yy, xx = torch.meshgrid(
            torch.linspace(-half_extent, half_extent, wave_resolution, dtype=torch.float32),
            torch.linspace(-half_extent, half_extent, wave_resolution, dtype=torch.float32),
            indexing="ij",
        )
        rr2 = xx**2 + yy**2
        aperture = (rr2 <= pupil_radius**2).float()
        self.register_buffer("xx", xx, persistent=False)
        self.register_buffer("yy", yy, persistent=False)
        self.register_buffer("rr2", rr2, persistent=False)
        self.register_buffer("aperture", aperture, persistent=False)

        fy = torch.fft.fftfreq(wave_resolution, d=self.wave_feature_size, dtype=torch.float32)
        fx = torch.fft.fftfreq(wave_resolution, d=self.wave_feature_size, dtype=torch.float32)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        self.register_buffer("fx2fy2", fx_grid.square() + fy_grid.square(), persistent=False)

        self.height_map_raw = nn.Parameter(self._initial_height_map(psf_init))
        self.W_qe = nn.Parameter(
            torch.tensor(
                [
                    [0.6737, 0.1069, 0.0501],
                    [0.1332, 0.9490, 0.2304],
                    [0.0409, 0.2324, 0.9220],
                ],
                dtype=torch.float32,
            )
        )

    def _initial_height_map(self, mode: str) -> torch.Tensor:
        if mode == "random":
            return 1.0e-2 * torch.randn((self.wave_resolution, self.wave_resolution), dtype=torch.float32)
        if mode == "gaussian":
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, self.wave_resolution, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, self.wave_resolution, dtype=torch.float32),
                indexing="ij",
            )
            return -2.0 * (xx**2 + yy**2)
        return torch.zeros((self.wave_resolution, self.wave_resolution), dtype=torch.float32)

    def height_map(self) -> torch.Tensor:
        return self.height_map_max * torch.sigmoid(self.height_map_raw)

    def noisy_height_map(self) -> torch.Tensor:
        height = self.height_map()
        if self.height_map_noise_std <= 0 or not self.training:
            return height
        noisy_height = height + self.height_map_noise_std * torch.randn_like(height)
        return torch.clamp(noisy_height, min=0.0, max=self.height_map_max)

    def quantized_height_map(self, height_map: torch.Tensor | None = None) -> torch.Tensor:
        height = self.height_map() if height_map is None else height_map
        if self.height_quantization_res <= 0:
            return height
        return torch.floor(height / self.height_quantization_res) * self.height_quantization_res

    def metadata(self) -> OpticalMetadata:
        return OpticalMetadata(
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            depth_layers=[float(x) for x in self.depth_layers.detach().cpu().tolist()],
            focus_distance=self.focus_distance,
            psf_size=self.psf_size,
            wavelengths=[float(x) for x in self.wavelengths],
            qe_matrix=self.effective_qe().detach().cpu().tolist(),
        )

    def _sensor_distance_for_focus(self) -> float:
        if not math.isfinite(self.focus_distance):
            return self.focal_length
        denom = (1.0 / self.focal_length) - (1.0 / self.focus_distance)
        if abs(denom) < 1e-12:
            return self.focal_length
        return 1.0 / denom

    def _phase_from_height_map(self, channel_idx: int, height_map: torch.Tensor | None = None) -> torch.Tensor:
        wavelength = self.wavelengths[channel_idx]
        refractive_index = self.refractive_indices[channel_idx]
        height = self.height_map() if height_map is None else height_map
        return (2.0 * math.pi / wavelength) * (refractive_index - 1.0) * height

    def _object_wave_phase(self, depth: float, channel_idx: int) -> torch.Tensor:
        wavelength = self.wavelengths[channel_idx]
        return math.pi * self.rr2 / (wavelength * max(depth, 1e-6))

    def _thin_lens_phase(self, channel_idx: int) -> torch.Tensor:
        if not self.use_thin_lens:
            return torch.zeros_like(self.rr2)
        wavelength = self.wavelengths[channel_idx]
        return -math.pi * self.rr2 / (wavelength * self.focal_length)

    def _propagate_fresnel(self, field: torch.Tensor, channel_idx: int) -> torch.Tensor:
        wavelength = self.wavelengths[channel_idx]
        transfer = torch.exp(-1j * math.pi * wavelength * self.sensor_distance * self.fx2fy2)
        padded = F.pad(field, (self.wave_resolution // 4,) * 4)
        transfer_padded = torch.exp(
            -1j
            * math.pi
            * wavelength
            * self.sensor_distance
            * (
                torch.fft.fftfreq(padded.shape[-2], d=self.wave_feature_size, dtype=torch.float32, device=field.device)
                .view(-1, 1)
                .square()
                + torch.fft.fftfreq(padded.shape[-1], d=self.wave_feature_size, dtype=torch.float32, device=field.device)
                .view(1, -1)
                .square()
            )
        )
        out = torch.fft.ifft2(torch.fft.fft2(padded) * transfer_padded)
        pad = self.wave_resolution // 4
        return out[pad:-pad, pad:-pad]

    @staticmethod
    def _center_crop_2d(image: torch.Tensor, target_size: int) -> torch.Tensor:
        if image.shape[-1] == target_size and image.shape[-2] == target_size:
            return image
        start_y = max((image.shape[-2] - target_size) // 2, 0)
        start_x = max((image.shape[-1] - target_size) // 2, 0)
        end_y = start_y + min(target_size, image.shape[-2])
        end_x = start_x + min(target_size, image.shape[-1])
        return image[start_y:end_y, start_x:end_x]

    def _sensor_sampling_factor(self) -> int:
        ratio = self.sensor_pixel_size / self.wave_feature_size
        return max(1, int(round(ratio)))

    def sampling_diagnostics(self) -> dict[str, float]:
        factor = self._sensor_sampling_factor()
        effective_sensor_pixel_size = factor * self.wave_feature_size
        ratio_error = abs(self.sensor_pixel_size - effective_sensor_pixel_size) / max(self.sensor_pixel_size, 1e-12)
        crop_size_wave = min(self.wave_resolution, self.psf_size * factor)
        crop_size_wave -= crop_size_wave % factor
        return {
            "wave_feature_size": float(self.wave_feature_size),
            "sensor_pixel_size": float(self.sensor_pixel_size),
            "sensor_sampling_factor": float(factor),
            "effective_sensor_pixel_size": float(effective_sensor_pixel_size),
            "sensor_sampling_ratio_error": float(ratio_error),
            "aperture_fill_ratio": float((2.0 * self.pupil_radius) / self.phase_mask_size),
            "sensor_crop_size_wave": float(crop_size_wave),
        }

    def height_map_stats(self) -> dict[str, float]:
        height = self.height_map().detach().cpu()
        return {
            "height_min": float(height.min()),
            "height_max": float(height.max()),
            "height_mean": float(height.mean()),
            "height_std": float(height.std()),
            "height_dynamic_range": float(height.max() - height.min()),
        }

    def _resample_to_sensor(self, intensity: torch.Tensor) -> torch.Tensor:
        factor = self._sensor_sampling_factor()
        crop_size = min(self.wave_resolution, self.psf_size * factor)
        crop_size -= crop_size % factor
        crop_size = max(crop_size, self.psf_size)
        cropped = self._center_crop_2d(intensity, crop_size)
        if factor > 1 and cropped.shape[-1] >= factor and cropped.shape[-2] >= factor:
            cropped = F.avg_pool2d(cropped.unsqueeze(0).unsqueeze(0), kernel_size=factor, stride=factor).squeeze(0).squeeze(0)
        if cropped.shape[-1] != self.psf_size or cropped.shape[-2] != self.psf_size:
            cropped = F.adaptive_avg_pool2d(cropped.unsqueeze(0).unsqueeze(0), (self.psf_size, self.psf_size)).squeeze(0).squeeze(0)
        return cropped

    def _compute_psf(self, depth: float, channel_idx: int, height_map: torch.Tensor | None = None) -> torch.Tensor:
        phase = (
            self._phase_from_height_map(channel_idx, height_map=height_map)
            + self._object_wave_phase(depth, channel_idx)
            + self._thin_lens_phase(channel_idx)
        )
        pupil = self.aperture * torch.exp(1j * phase)
        field = self._propagate_fresnel(torch.fft.ifftshift(pupil), channel_idx)
        intensity = field.real.square() + field.imag.square()
        # Shift the PSF peak back to the image center so spatial convolution
        # interprets the kernel origin consistently with DirectPSF kernels.
        intensity = torch.fft.fftshift(intensity)
        # Crop to the central sensor field-of-view and integrate onto the
        # coarser camera pixel grid rather than using a direct global resize.
        intensity = self._resample_to_sensor(intensity)
        return intensity / (intensity.sum() + 1e-8)

    def _psf_from_height_map(self, height_map: torch.Tensor) -> torch.Tensor:
        psfs = []
        for depth in self.depth_layers:
            channel_psfs = []
            depth_value = float(depth.item())
            for channel_idx in range(self.num_channels):
                channel_psfs.append(self._compute_psf(depth_value, channel_idx, height_map=height_map))
            psfs.append(torch.stack(channel_psfs, dim=0))
        return torch.stack(psfs, dim=0)

    def effective_psf(self) -> torch.Tensor:
        sampled_height_map = self.noisy_height_map()
        return self._psf_from_height_map(sampled_height_map)

    def continuous_psf(self) -> torch.Tensor:
        return self._psf_from_height_map(self.height_map())

    def quantized_psf(self) -> torch.Tensor:
        return self._psf_from_height_map(self.quantized_height_map())

    def psf_for_mode(self, mode: str = "default") -> torch.Tensor:
        if mode == "default":
            return self.effective_psf()
        if mode == "continuous":
            return self.continuous_psf()
        if mode == "quantized":
            return self.quantized_psf()
        raise ValueError(f"Unsupported PSF mode: {mode}")

    def psf_for_depth(self, depth_idx: int) -> torch.Tensor:
        return self.effective_psf()[depth_idx]

    def psf_slice(self, depth_idx: int, channel_idx: int) -> torch.Tensor:
        return self.psf_for_depth(depth_idx)[channel_idx]

    def regularization_loss(self) -> torch.Tensor:
        loss = self.height_map_raw.new_tensor(0.0)
        if self.laplace_reg > 0:
            kernel = torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]],
                dtype=self.height_map_raw.dtype,
                device=self.height_map_raw.device,
            ).view(1, 1, 3, 3)
            laplace = F.conv2d(
                self.height_map().view(1, 1, self.wave_resolution, self.wave_resolution),
                kernel,
                padding=1,
            )
            loss = loss + self.laplace_reg * laplace.square().mean()

        if self.psf_edge_reg > 0:
            psf = self.effective_psf()
            border = max(1, self.psf_size // 16)
            edge_mask = torch.zeros((self.psf_size, self.psf_size), dtype=torch.bool, device=psf.device)
            edge_mask[:border, :] = True
            edge_mask[-border:, :] = True
            edge_mask[:, :border] = True
            edge_mask[:, -border:] = True
            edge_energy = psf[..., edge_mask].sum(dim=-1)
            loss = loss + self.psf_edge_reg * edge_energy.mean()

        if self.quantization_reg > 0 and self.height_quantization_res > 0:
            height = self.height_map()
            quantized = self.quantized_height_map(height)
            loss = loss + self.quantization_reg * (height - quantized).square().mean()

        return loss
