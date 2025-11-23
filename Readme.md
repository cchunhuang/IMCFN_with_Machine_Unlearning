# IMCFN with Machine Unlearning

This project combines the approaches from two papers:
1. "IMCFN: Image-based malware classification using fine-tuned convolutional neural network architecture" by Danish Vasan et al.
2. "Machine Unlearning" by Lucas Bourtoule et al.

It implements an image-based malware classification system using fine-tuned CNN architecture, enhanced with machine unlearning capabilities to allow for efficient removal of specific training data points.

## Project Description

This project implements an Unlearnable Image-based Malware Classification using Fine-tuned Convolutional Neural Network (Unlearnable IMCFN). It's designed to detect malware using image classification techniques while allowing for the "unlearning" or removal of specific data points from the trained model.

Key features:
- Converts binary files to images for classification
- Uses a VGG16-based neural network for image classification
- Implements a shard-based training approach for efficient unlearning
- Provides functionalities for training, prediction, and unlearning
- Utilizes a configuration-driven approach for easy customization

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Setup](#quick-setup)
3. [Manual Setup](#manual-setup)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Configuration](#configuration)
7. [Model Architecture](#model-architecture)
8. [Data Processing](#data-processing)
9. [Training Process](#training-process)
10. [Unlearning Process](#unlearning-process)
11. [Key Methods Reference](#key-methods-reference)
12. [Output Structure](#output-structure)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)
15. [License](#license)
16. [References](#references)

## Project Structure

```
IMCFN/
├── src/                      # Source code
│   ├── main.py              # Entry point
│   ├── UnlearnableIMCFN.py  # Main implementation
│   ├── IMCFN.py             # Base classifier
│   ├── VGG16.py             # Neural network model
│   ├── ImageGenerator.py    # Binary-to-image converter
│   ├── Logger.py            # Logging setup
│   ├── config.json          # Configuration file
│   └── logging_config.json  # Logging configuration
├── requirements_*.txt       # Python dependencies
└── Readme.md               # This file
```

## Quick Setup

Running autobuild.sh
1. Ensure Execute Permission: Grant execute permissions to the script:
   ```
   chmod +x autobuild.sh
   ```
2. Run the Script:
   ```
   ./autobuild.sh
   ```
3. After the script completes, activate the environment:
   ```
   conda activate IMCFN
   ```

What autobuild.sh Does?
1. Dependency Check: Ensures wget, curl, and git are installed. Installs them if missing.
2. Anaconda Installation: Checks for Anaconda and installs the latest version if not found.
3. Environment Creation: Creates a conda environment named IMCFN with Python 3.11.5.
4. Repository Cloning: Clones the IMCFN project repository. If a previous directory exists, prompts for deletion.
5. Requirements Installation: Installs required Python packages from requirements_cpu.txt.


## Manual Setup

This project was developed using Python 3.11.5. It is strongly recommended to use this version for optimal compatibility and performance.

1. Ensure you have Python 3.11.5 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3115/).

2. Clone the repository:
   ```bash
   git clone https://github.com/cchunhuang/IMCFN_with_Machine_Unlearning.git
   cd IMCFN_with_Machine_Unlearning
   ```

3. Choose the appropriate requirements file based on your hardware:
   - `requirements_cpu.txt`: For CPU-only systems
   - `requirements_cuda118.txt`: For systems with CUDA 11.8
   - `requirements_cuda121.txt`: For systems with CUDA 12.1

   You can also choose a different PyTorch version that suits your needs.

4. Install the required dependencies:
   ```bash
   pip install -r requirements_[your_choice].txt
   ```

   Replace `[your_choice]` with either `cpu`, `cuda118`, or `cuda121` based on your selection.

5. Prepare your dataset:
   - Place binary files in the dataset folder (e.g., `./dataset/`)
   - Create a CSV label file with columns: filename, label, type (train/test)
   - Update the paths in `config.json` to match your setup

## Usage

This project can be used in three ways:
1. By running `main.py`
2. By directly calling functions from `UnlearnableIMCFN.py`
3. By directly calling functions from `IMCFN.py`

Here are examples of how to use the project:

### Using main.py

To use the project via `main.py`, simply run the script with the appropriate configuration file:

```bash
python main.py [config_path]
```

If no configuration file is provided, it will use the default `./config.json`. The script will automatically determine the mode (train, predict, or unlearn) based on the `action` field in the configuration file.

Supported actions:
- `"train"`: Train the model with sharded approach
- `"predict"`: Make predictions using ensemble voting
- `"unlearn"`: Perform machine unlearning on specific data samples

### Using UnlearnableIMCFN Directly

You can use the `UnlearnableIMCFN` class directly in your Python scripts:

```python
from UnlearnableIMCFN import UnlearnableIMCFN

# Initialize with a configuration file
config_path = './config.json'
unlearnable_imcfn = UnlearnableIMCFN(config_path)

# Training with sharded approach
unlearnable_imcfn.trainModel()

# Prediction
predictions = unlearnable_imcfn.predict()

# Unlearning
unlearnable_imcfn.unlearn()

# Train specific shards (for unlearning)
unlearnable_imcfn.trainModel(shard_list=[0, 2], start_slice=[1, 0])
```

### Using IMCFN Directly

You can use the base `IMCFN` class for basic malware classification:

```python
from IMCFN import IMCFN

# Initialize with a configuration file
config_path = './config.json'
imcfn = IMCFN(config_path)

# Vectorize the data (convert binary to images)
imcfn.get_vector()

# Train the model
result = imcfn.get_model(action="train")

# Make predictions
predictions = imcfn.get_prediction(data_type='test')

# Get labels from CSV file
labels = imcfn.getLabel('dataset.csv', data_type='train')
```

Note: The base `IMCFN` class doesn't support the sharded training approach used for machine unlearning.

## File Structure

### Core Files
- `main.py`: Entry point for running the UnlearnableIMCFN system. Handles command-line arguments and dispatches to train, predict, or unlearn actions.
- `UnlearnableIMCFN.py`: Main class implementing the unlearnable IMCFN with sharded and sliced training approach for machine unlearning.
- `IMCFN.py`: Base class implementing the IMCFN (Image-based Malware Classification using Fine-tuned Convolutional Neural Network). Inherits from `MalwareClassifier` base class.
- `ImageGenerator.py`: Converts malware binary files to image representations using colormap visualization (jet colormap).
- `VGG16.py`: Implements a customized VGG16 neural network with transfer learning for malware classification.
- `Logger.py`: Sets up logging system for the project using MalwareClassifier's logging infrastructure with project-specific configuration.

### Configuration Files
- `config.json`: Main configuration file for the system (used by both IMCFN and UnlearnableIMCFN).
- `logging_config.json`: Configuration file for logging settings.

## Configuration

The project uses a single JSON configuration file (`config.json`) with the following structure:

### Configuration Sections

1. **`file`**: File paths configuration
   - `label`: Path to CSV file containing dataset labels
   - `pretrained`: Path to pretrained model or "DEF" for default VGG16 weights
   - `logging_config`: Path to logging configuration JSON file
   - `model`: Path to save trained model (auto-generated for UnlearnableIMCFN)
   - `log`: Path to log file
   - `position`: Position tracking file for sharded training
   - `subdetector_name`: Subdetector naming file for sharded models. **Modify the folder of `subdetector_name.csv` to select specific pretrained model!**
   - `score`: Path to save training scores
   - `result`: Path to save prediction results

2. **`folder`**: Directory paths configuration
   - `dataset`: Directory containing binary files to classify
   - `vector`: Directory to save generated images (if save_image is enabled)
   - `model`: Directory to save trained models
   - `predict`: Directory to save prediction results
   - `label`: Directory containing label files
   - `log`: Directory for log files

3. **`params`**: Model parameters and settings
   - `mode`: Operation mode (e.g., "detection")
   - `feature`: Feature extraction parameters (empty, not used in current implementation)
   - `vector`: Vectorization settings
     - `save_image`: Whether to save generated images (true/false)
   - `model`: Model training parameters
     - `model_name`: Neural network architecture (e.g., "VGG16")
     - `batch_size`: Batch size for training (default: 32)
     - `learning_rate`: Learning rate for SGD optimizer (default: 5e-6)
     - `rotation`: Range for random rotation in degrees [min, max] (e.g., [0, 0] for no rotation)
     - `width_shift`: Horizontal shift range as fraction of total width (e.g., 0.0)
     - `height_shift`: Vertical shift range as fraction of total height (e.g., 0.0)
     - `zoom`: Scale range for random zoom [min, max] (e.g., [1.0, 1.0] for no zoom)
     - `shear`: Shear range [x_min, x_max, y_min, y_max] (e.g., [0, 0, 0, 0] for no shear)
     - `fill`: Pixel fill value for areas outside boundaries (null for default)
     - `horizontal_flip`: Probability of random horizontal flip (0-1)
     - `train_ratio`: Ratio of training data to validation data (default: 0.8)
     - `epochs`: Number of training epochs (default: 30)
     - `shard`: Number of shards for distributed training (default: 1)
     - `slice`: Number of slices per shard (default: 1)
     - `print_information`: Additional information to print/log during training
     - `overwrite`: Whether to overwrite existing model files (default: false)
   - `predict`: Prediction parameters (empty in current implementation)

4. **`action`**: Operation to perform
   - `"train"`: Train the model
   - `"predict"`: Make predictions on dataset
   - `"unlearn"`: Unlearn data

## Model Architecture

The project uses a modified VGG16 architecture for malware classification:

### Base Architecture
- Pre-trained VGG16 model from torchvision
- Input size: 224x224x3 (RGB images)
- Transfer learning approach: Load pre-trained ImageNet weights

### Customization
- Custom classifier layers replace the original VGG16 classifier
- Specific layers are unfrozen for fine-tuning on malware data
- Multi-class classification output (number of classes = number of malware families + 1 for benignware)

### Training Configuration
- Optimizer: SGD (Stochastic Gradient Descent)
- Loss function: CrossEntropyLoss
- Learning rate: Configurable (default: 5e-6)
- Batch size: Configurable (default: 32)
- Supports loading checkpoints for continued training or inference

### Key Features
- Automatic device selection (CUDA if available, otherwise CPU)
- Data augmentation during training
- Train/validation split for model evaluation
- Batch normalization and dropout for regularization
- Support for saving and loading model weights

## Data Processing

### Binary to Image Conversion
- Binary malware files are converted to images using the `ImageGenerator` class
- Uses colormap visualization (matplotlib's jet colormap) for binary-to-image conversion
- Image width is automatically determined based on file size:
  - ≤10KB: 32px width
  - ≤30KB: 64px width
  - ≤60KB: 128px width
  - ≤100KB: 256px width
  - ≤200KB: 384px width
  - ≤500KB: 512px width
  - ≤1000KB: 768px width
  - >1000KB: 1024px width
- Binary data is read as uint8 array and reshaped into 2D array based on calculated width
- Colormap is applied to convert 2D grayscale to 4-channel RGBA (224x224x4)
- Images are resized to 224x224 pixels using OpenCV for VGG16 input
- RGB channels are used (first 3 channels [:, :, :3] of the RGBA output)

### Data Augmentation
- Configurable data augmentation techniques:
  - Random rotation
  - Width and height shift
  - Zoom/scale transformation
  - Shear transformation
  - Horizontal flip
  - Custom fill mode for transformed areas

### Dataset Structure
- Dataset labels are stored in CSV format with columns:
  - File name
  - Label (malware family or "benignware")
  - Type ("train" or "test")
- Dataset is split into training and validation sets based on `train_ratio`
- Supports separate test set for evaluation

## Training Process

### Basic Training (IMCFN)
- Uses transfer learning with pre-trained VGG16 model
- Fine-tunes specific layers for malware classification
- Uses SGD optimizer with configurable learning rate
- Uses CrossEntropyLoss for multi-class classification
- Automatically evaluates on test set if available
- Calculates metrics: accuracy, precision, recall, F1 score
- Saves model checkpoints with timestamp
- Logs training progress and metrics

### Sharded Training (UnlearnableIMCFN)
- Implements a shard-based and slice-based training approach
- Data is distributed across multiple shards and slices:
  - Each shard contains multiple slices of data
  - Batch size determines data distribution granularity
  - Training is performed incrementally on slices
- Enables efficient machine unlearning by retraining specific shards
- Supports resuming training from specific shard and slice
- Maintains separate models for each shard-slice combination
- Uses SEED value (42) for reproducible data shuffling
- Saves position tracking and subdetector naming information

## Unlearning Process

The machine unlearning capability is achieved through the sharded training approach:

1. **Data Location**: Identify which shards and slices contain the data to be unlearned
2. **Selective Retraining**: Only retrain the affected shards from the appropriate slice
3. **Model Preservation**: Other shards remain unchanged, preserving learned information
4. **Efficient Updates**: No need to retrain the entire model from scratch

### How to Perform Unlearning

```python
# Example: Unlearn data in shard 0 and 2, retraining from slice 1 and 0 respectively
unlearnable_imcfn.trainModel(shard_list=[0, 2], start_slice=[1, 0])
```

**Note**: The unlearning functionality is fully implemented in `main.py` and can be triggered by:
1. Setting `action` to `"unlearn"` in the configuration file
2. Ensuring `file.label` points to a CSV file containing the samples to be unlearned
3. The `unlearn()` method will:
   - Identify which shards and slices contain the data to be unlearned
   - Remove those samples from the input data
   - Retrain only the affected shards from the appropriate slices
   - Save updated position and subdetector name files

## Key Methods Reference

### IMCFN Class Methods

- `get_feature()`: Not required for IMCFN (logs that feature extraction is not needed)
- `get_vector()`: Convert binary files to image representations and create label indices
- `get_model(action)`: Train model (action="train") or prepare for inference (action="predict"). Returns training results dictionary if training.
- `get_prediction(data_type)`: Make predictions on specified data type. Returns dictionary mapping file names to predicted labels.
- `getLabel(file_path, full_path, data_type)`: Load labels from CSV file. Returns dictionary mapping paths/names to labels.
- `getFilePath(file_path, data_type)`: Get file paths filtered by data type. Returns list of full paths.
- `getFileName(file_path, data_type)`: Get file names filtered by data type. Returns list of file names.

### UnlearnableIMCFN Class Methods

- `trainModel(shard_list, start_slice)`: Train model with sharded and sliced approach. Supports selective retraining of specific shards and slices.
- `predict()`: Make predictions using ensemble voting across all shard models. Returns list of dictionaries with file names and predictions.
- `unlearn()`: Remove specific training samples from the model by retraining affected shards and slices.
- `savePosition()`: Save data position (shard and slice index) tracking for each sample to CSV file.
- `getPosition()`: Load data position tracking from CSV file.
- `saveSubdetectorLabel(data)`: Save labels for current subdetector training batch to CSV.
- `saveSubdetectorName(model_name)`: Save model file paths for all subdetectors to CSV.
- `getSubdetectorName()`: Load model file paths for all subdetectors from CSV.
- `getLabel(path, data_type)`: Load labels with optional data type filtering. Returns list of [file_name, label] pairs.
- `getFileName(data_type)`: Get file names with optional data type filtering. Returns list of file names.

## Output Structure

When training, the system creates the following structure:

```
output/
├── model/
│   └── YYYYMMDD_HHMM/              # Timestamp of training session
│       ├── shard0/
│       │   ├── slice0.pth           # Model checkpoint for shard 0, slice 0
│       │   ├── slice1.pth
│       │   ├── log.txt              # Training logs for this shard
│       │   └── ...
│       ├── shard1/
│       │   └── ...
│       ├── subdetector_label.csv    # Labels for current training batch
│       ├── subdetector_config.json  # Configuration for subdetectors
│       ├── position.csv             # Data position tracking
│       ├── subdetector_name.csv     # Subdetector model names
│       ├── score.json               # Training scores and metrics
│       ├── predict_result.json      # Prediction results (if predict action)
│       └── loging.log               # Main training logs
└── predict/
    └── result.json                  # Prediction results (if overwrite=true)
```

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure all paths in `config.json` are correct and point to existing files/directories
2. **CUDA out of memory**: Reduce `batch_size` in configuration
3. **Image generation fails**: Check that binary files are accessible and not corrupted
4. **Model loading errors**: Verify that `pretrained` path points to a valid model file or use "DEF" for default weights

### Logging

- All operations are logged to the file specified in `config.file.log`
- Logging configuration can be customized in `logging_config.json`
- Log level and format can be adjusted for debugging

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate documentation.

## License

See LICENSE file for details.

## References

1. Vasan, D., Alazab, M., Wassan, S., Naeem, H., Safaei, B., & Zheng, Q. (2020). IMCFN: Image-based malware classification using fine-tuned convolutional neural network architecture. Computer Networks, 171, 107138.
2. Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, B., ... & Papernot, N. (2021). Machine unlearning. In 2021 IEEE Symposium on Security and Privacy (SP) (pp. 141-159). IEEE.

## Author

cchunhuang - 2024

