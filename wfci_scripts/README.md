Usage: 
Check common_commands.md for concrete examples
To run motion correction (and crop the data around the brain to exclude empty pixels): 

```bash
python motion_correct_and_crop \
    path=/path/to/datafile \
    output_file=/path/outputname.npz \
    crop_height_start=<start_val> \
    crop_height_end=<end_val> \
    crop_width_start=<start_val> \
    crop_width_end=<end_val> \
    max_rigid_shift=3
```
This saves out the motion corrected data + the shifts at each frame

To run the training loop for training a neural network denoiser: 
```bash
python train_blindspot_net.py \
    npz_path=/path/to/moco_data.npz \
    output_file= /file/to/save.npz \
    block_size_dim1=32 \
    block_size_dim2=32 \
    background_rank=0 \
    max_components=20 \
    max_consecutive_failures=1 \
    spatial_avg_factor=1 \
    temporal_avg_factor=1 \
    device=cpu \
    frame_batch_size=1024 \
    epochs=5 \
    learning_rate=1e-4
 ```
This will output .npz file

To run compression and denoising on imaging data using a pre-trained network:

```bash
python compress_and_denoise.py \
    input=/path/to/data.npz \
    output=/path/to/output.npz \
    block_size_dim1=32 \
    block_size_dim2=32 \
    background_rank=15 \
    max_components=20 \
    max_consecutive_failures=1 \
    spatial_avg_factor=1 \
    temporal_avg_factor=1 \
    device=cpu \
    frame_batch_size=1024 \
    neural_network=/path/to/network.npz
```