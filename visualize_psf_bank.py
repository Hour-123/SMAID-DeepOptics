import argparse
import os

import numpy as np
import torch

from utils import img_io


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a saved PSF bank.")
    parser.add_argument("--input", required=True, help="Path to psf_bank.npy")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    psf_bank = np.load(args.input)
    psf_tensor = torch.from_numpy(psf_bank)
    os.makedirs(args.output_dir, exist_ok=True)
    img_io.save_psf_bank(psf_tensor, args.output_dir)
    img_io.save_full_psf_visualizations(psf_tensor, args.output_dir)
    print(f"Saved PSF visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
