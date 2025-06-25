import os

import sys
import masknmf
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

from masknmf import display

@hydra.main()
def train_denoiser(cfg: DictConfig) -> None:
    input_file = os.path.abspath(cfg.npz_path)
    if not os.path.exists(input_file):
        raise ValueError(f"the path {input_file} does not seem to exist")
    input_file = np.load(input_file)
    my_data = input_file['moco']
    my_shifts = input_file['shifts']
    max_shifts = np.ceil(np.amax(np.abs(my_shifts), axis = 0))
    max_shifts = [int(i) for i in max_shifts]

    display(f"The max height shift was at most {max_shifts[0]} and the max width was at most {max_shifts[1]}."
            "Cropping the borders appropriately")
    my_data = my_data[:, max_shifts[0]:-1*max_shifts[0], max_shifts[1]:-1*max_shifts[1]]
    display(f"post crop the shape is {my_data.shape}")
    block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

    device = cfg.device
    if device == 'cpu':
        display("Running PMD to generate training data on CPU")

    pmd_obj = masknmf.compression.pmd_decomposition(my_data,
                                                    block_sizes,
                                                    my_data.shape[0],
                                                    max_components=cfg.max_components,
                                                    max_consecutive_failures=cfg.max_consecutive_failures,
                                                    temporal_avg_factor=cfg.temporal_avg_factor,
                                                    spatial_avg_factor=cfg.spatial_avg_factor,
                                                    background_rank=cfg.background_rank,
                                                    device=device,
                                                    frame_batch_size=cfg.frame_batch_size)

    v = pmd_obj.v.cpu()
    trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                                                   max_epochs=cfg.epochs,
                                                                                   batch_size=128,
                                                                                   learning_rate=cfg.learning_rate)
    save_path = os.path.abspath(os.path.join(cfg.output_file))
    np.savez(save_path, model=trained_model)
    display("Results saved successfully")


if __name__ == "__main__":
    config_dict = {
        'npz_path': '/path/to/data/',
        'output_file': 'output/file/path',
        'block_size_dim1': 32,
        'block_size_dim2': 32,
        'background_rank': 0,
        'max_components': 20,
        'max_consecutive_failures': 1,
        'spatial_avg_factor': 1,
        'temporal_avg_factor': 1,
        'device': 'cpu',
        'frame_batch_size': 1024,
        ## For training the network:
        'epochs': 5,
        'learning_rate': 1e-4,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    train_denoiser(cfg)