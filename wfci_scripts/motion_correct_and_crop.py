import os
import sys
import sys
import masknmf
from masknmf import display
import numpy as np

import h5py
from masknmf.utils import display
from typing import *
import pathlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

def load_file(file_path: str) -> np.ndarray:
    """
    Basic routine to load file
    """
    with h5py.File(file_path, 'r') as f:
        # Print top-level keys (e.g., ['data'])
        # Load the 'blue' dataset
        blue = np.array(f['data']['blue'])
        display(f"data shape {blue.shape}")
        return blue
    
@hydra.main()
def motion_correct_and_crop(cfg: DictConfig) -> None:
    device = cfg.device
    file_path = os.path.abspath(cfg.path)

    if not os.path.exists(file_path):
        raise ValueError(f"the path {file_path} does not seem to exist")

    raw_data = load_file(file_path)

    height_crop_info = cfg.crop_height_start, cfg.crop_height_end
    width_crop_info = cfg.crop_width_start, cfg.crop_width_end


    if height_crop_info[0] is not None and height_crop_info[1] is not None and width_crop_info[0] is not None and width_crop_info[1] is not None:
        display("Spatially cropping the dataset")
        slice_height = slice(height_crop_info[0], height_crop_info[1])
        slice_width = slice(width_crop_info[0], width_crop_info[1])
        raw_data = raw_data[:, slice_height, slice_width]
        display(f"Cropped data has shape {raw_data.shape}")
    else:
        display("No cropping performed")

    display("Motion Correction")

    max_rigid_shifts = [cfg.max_shifts_height, cfg.max_shifts_width]
    template = None
    rigid_strategy = masknmf.RigidMotionCorrection(max_rigid_shifts, template=template)

    rigid_strategy = masknmf.compute_template(raw_data,
                                              rigid_strategy,
                                              device=device,
                                              batch_size=cfg.frame_batch_size)

    moco_results = masknmf.RegistrationArray(raw_data,
                                             rigid_strategy,
                                             device=device,
                                             batch_size=cfg.frame_batch_size)

    frames_to_access = slice(0, raw_data.shape[0])

    # moco_shifts is a (num_frames, 2) array. moco_shifts[:, 0] gives you vertical shifts, moco_shifts[:, 1] gives you horizontal shifts.
    moco_stack, moco_shifts = [i.cpu().numpy() for i in moco_results.index_frames_tensor(frames_to_access)]

    display("Saving results")
    if cfg.output_file is None:
        output_file = os.path.abspath("./motion_correction_results.npz")
        display(f"No output was provided, saving to {output_file}")
    else:
        output_file = os.path.abspath(cfg.output_file)
    np.savez(output_file, moco = moco_stack, shifts = moco_shifts)
    display("Results Saved")
    
    

if __name__ == "__main__":
    config_dict = {
        'path': '/path/to/data/',
        'output_file': None,
        'crop_height_start': None,
        'crop_height_end': None,
        'crop_width_start': None,
        'crop_width_end': None,
        'max_shift_height': 3,
        'max_shift_width': 3,
        'device': 'cpu',
        'frame_batch_size': 1024,
    }

    cfg = OmegaConf.create(config_dict)
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)

    motion_correct_and_crop(cfg)