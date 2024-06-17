# Texture Optimizer

## Overview

The Texture Optimizer is a deep learning model based on PyTorch3D, designed to optimize textures for 3D models based on given ground truth images.

## Requirements

- Python 3.10
- PyTorch 2.1.0
- PyTorch3D 0.7.5
- OpenCV python
- tqdm
- tensorboard


## Training Data

The training data should be organized in a specific directory structure:

```
data_dir/
    poses.txt
    mesh/
        <mesh_name>.obj
    images/
        001.png
        002.png
        ...
```

- `poses.txt` saves the camera poses of images in KITTI format.
- `mesh/` contains the 3D mesh file in `.obj` format.
- `images/` contains the ground truth images used for training. These images should be named sequentially.

## Running the Training Script

To start training, use the following command:

```bash
python train.py --data_dir <path_to_data_dir> --save_dir <path_to_save_dir> --mesh_name <mesh_file_name> --camera_param_type <camera_params_type> --device cuda --epochs 10 --lr 0.01 --texture_size 512
```

### Arguments

- `--data_dir`: Path to the training data directory.
- `--save_dir`: Path where the trained model and outputs will be saved.
- `--mesh_name`: Name of the mesh file (without extension).
- `--camera_param_type`: Type of camera parameterization (default: "fov").
- `--device`: Device to run the training on ("cuda" for GPU or "cpu").
- `--epochs`: Number of epochs to train (default: 10).
- `--lr`: Learning rate (default: 0.01).
- `--texture_size`: Size of the texture image (default: 512).

## Training Process

The training process involves the following steps:

1. **Initialization**: The script initializes the model, optimizer, and data loaders based on the provided arguments.
2. **Training Loop**: For each epoch, the model is trained on the training dataset. Losses are calculated using the L1 loss between the rendered images and the ground truth images. The losses and rendered images are logged to TensorBoard.
3. **Validation**: After each epoch, the model is evaluated on the validation dataset, and validation losses and images are logged to TensorBoard.
4. **Saving**: At the end of training, the optimized texture is saved as `texture.png` in the specified save directory.

## Monitoring Training

Training progress can be monitored using TensorBoard. To launch TensorBoard, run:

```bash
tensorboard --logdir <path_to_save_dir>
```

Navigate to the provided URL in your web browser to view the training and validation losses, rendered images, and the optimized texture over time.
