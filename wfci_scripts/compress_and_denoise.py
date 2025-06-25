import os
import sys
import sys
import masknmf
from masknmf import display
import tifffile
import torch
import numpy as np

import matplotlib.pyplot as plt
import time
from typing import *
import pathlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main()
def compress_and_denoise(cfg: DictConfig) -> None:
    input_filepath = os.path.abspath(cfg.input)
    if not os.path.exists(input_filepath):
        raise ValueError(f"the path {input_filepath} does not seem to exist")
    input_file = np.load(input_filepath)
    my_data = input_file['moco']
    my_shifts = input_file['shifts']
    max_shifts = np.ceil(np.amax(np.abs(my_shifts), axis=0))
    max_shifts = [int(i) for i in max_shifts]

    display(f"The max height shift was at most {max_shifts[0]} and the max width was at most {max_shifts[1]}."
            "Cropping the borders appropriately")
    my_data = my_data[:, max_shifts[0]:-1 * max_shifts[0], max_shifts[1]:-1 * max_shifts[1]]
    display(f"post crop the shape is {my_data.shape}")

    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    device = cfg.device
    if device == 'cpu':
        display("Running PMD to generate training data on CPU")

    if cfg.neural_network is not None:
        net_path = os.path.abspath(cfg.neural_network)
        trained_model = np.load(net_path, allow_pickle=True)['model'].item()
        curr_temporal_denoiser = masknmf.compression.PMDTemporalDenoiser(trained_model)
    else:
        curr_temporal_denoiser = None

    pmd_denoised = masknmf.compression.pmd_decomposition(my_data,
                                                         block_sizes,
                                                         my_data.shape[0],
                                                         max_components=cfg.max_components,
                                                         max_consecutive_failures=cfg.max_consecutive_failures,
                                                         temporal_avg_factor=cfg.temporal_avg_factor,
                                                         spatial_avg_factor=cfg.spatial_avg_factor,
                                                         background_rank=cfg.background_rank,
                                                         device=device,
                                                         temporal_denoiser=curr_temporal_denoiser,
                                                         frame_batch_size=cfg.frame_batch_size)

    pmd_no_denoise = masknmf.compression.pmd_decomposition(my_data,
                                                           block_sizes,
                                                           my_data.shape[0],
                                                           max_components=cfg.max_components,
                                                           max_consecutive_failures=cfg.max_consecutive_failures,
                                                           temporal_avg_factor=cfg.temporal_avg_factor,
                                                           spatial_avg_factor=cfg.spatial_avg_factor,
                                                           background_rank=cfg.background_rank,
                                                           device=device,
                                                           temporal_denoiser=None,  # Turn off denoiser
                                                           frame_batch_size=cfg.frame_batch_size)

    display(
        f"Processing complete. The rank of PMD with denoiser is {pmd_denoised.pmd_rank}. The rank of PMD without denoiser is {pmd_no_denoise.pmd_rank}")

    outdir = os.path.abspath(cfg.output)
    if os.path.isdir(outdir):
        output_location = os.path.join(os.path.abspath(cfg.output), "pmd_results.npz")
    else:
        output_location = cfg.output

    # From this, it is easy to load the results into a notebook, visualize things, etc.sl
    np.savez(output_location,
             pmd_denoise=pmd_denoised,
             pmd_no_denoise=pmd_no_denoise,
             raw_path=input_filepath)
    display("Results saved")


if __name__ == "__main__":
    config_dict = {
        'input': '/path/to/data.npz',
        'output': '/path/to/output.npz',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'background_rank': 15,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'device': 'cpu',
        'frame_batch_size': 1024,
        'neural_network': None
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    compress_and_denoise(cfg)