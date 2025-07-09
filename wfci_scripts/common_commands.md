Run Moco: 

```
python motion_correct_and_crop.py path=/media/app2139/Extreme\ SSD/cm9999_9\ Registration/registered_runD.mat output_file=./results/moco_results.npz
```

Run Training:

```
python train_blindspot_net.py npz_path=./results/motion_correction_results.npz output_file=./results/neural_net.npz device=cuda
```

Run Compression

```
python compress_and_denoise.py input=./results/motion_correction_results.npz  output=./results/compression_t1.npz neural_network=./results/neural_net.npz device=cuda
```