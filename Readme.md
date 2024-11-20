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

1. [Quick Setup](#quick-setup)
2. [Manual Setup](#manual-setup)
3. [Usage](#usage)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Model Architecture](#model-architecture)
7. [Data Processing](#data-processing)
8. [Training Process](#training-process)
9. [Unlearning Process](#unlearning-process)

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
   ```
   git clone [repository_url]
   cd [repository_name]
   ```

3. Choose the appropriate requirements file based on your hardware:
   - `requirements_cpu.txt`: For CPU-only systems
   - `requirements_cuda118.txt`: For systems with CUDA 11.8
   - `requirements_cuda121.txt`: For systems with CUDA 12.1

   You can also choose a different PyTorch version that suits your needs.

4. Install the required dependencies:
   ```
   pip install -r requirements_[your_choice].txt
   ```

   Replace `[your_choice]` with either `cpu`, `cuda118`, or `cuda121` based on your selection.

5. Set up the project structure as defined in the configuration file.

## Usage

This project can be used in three ways:
1. By running `main.py`
2. By directly calling functions from `UnlearnableIMCFN.py`
3. By directly calling functions from `IMCFN.py`

Here are examples of how to use the project:

### Using main.py

To use the project via `main.py`, simply run the script with the appropriate configuration file:

```
python main.py path/to/your/config.json
```

The script will automatically determine the mode (train, predict, or unlearn) based on the configuration file.

### Using UnlearnableIMCFN.py

You can use the `UnlearnableIMCFN` class directly in your Python scripts:

```python
from UnlearnableIMCFN import UnlearnableIMCFN

# Initialize with a configuration file
config_path = 'path/to/your/dataToPython.json'
uv = UnlearnableIMCFN(config_path)

# Training
uv.trainModel()

# Prediction
predictions = uv.predict()

# Unlearning
uv.unlearn()
```

Make sure to set the appropriate flags in your configuration file for the desired operation mode.

### Using IMCFN.py

You can use the `IMCFN` class directly in your Python scripts for basic malware classification without unlearning capabilities:

```python
from IMCFN import IMCFN

# Initialize with a configuration file
config_path = 'path/to/your/config.json'
imcfn = IMCFN(config_path)

# Vectorize the data (convert binary to images)
imcfn.vectorize()

# Train the model
result = imcfn.model(training=True)

# Make predictions
predictions = imcfn.predict()

# Get labels
labels = imcfn.getLabel('path/to/label/file.csv')
```

Note that when using IMCFN directly, you won't have access to the unlearning functionality provided by UnlearnableIMCFN.

## File Structure

- `UnlearnableIMCFN.py`: Main class implementing the unlearnable IMCFN.
- `IMCFN.py`: Implements the base IMCFN (Image-based Malware Classification using Fine-tuned Convolutional Neural Network).
- `ImageGenerator.py`: Handles the conversion of binary files to images.
- `VGG16.py`: Implements the VGG16 neural network model.
- `Logger.py`: Sets up logging for the project.
- `main.py`: Entry point for running the project.
- `UnlearningConfig.py`: Handles the configuration parsing and setup.
- `logging_config.json`: Configuration file for logging.
- `dataToPython.json`: Configuration file for UnlearnableIMCFN.
- `config.json`: Configuration file for IMCFN.

## Configuration

The project uses JSON configuration files. There are two main configuration files:

1. `dataToPython.json`: Used by UnlearnableIMCFN. It includes:
   - `path`: Specifies file paths (input files, logging config, etc.)
   - `folder`: Specifies various folder paths (dataset, model, etc.)
   - `model`: Sets model parameters (batch size, learning rate, etc.)
   - `classify`, `train`, `predict`, `unlearn`: Flags for different operation modes

2. `config.json`: Used by IMCFN. It includes similar sections as `dataToPython.json`, but may have some differences specific to IMCFN without unlearning capabilities.

Ensure you have the correct configuration file for the class you're using (UnlearnableIMCFN or IMCFN).

## Model Architecture

The project uses a modified VGG16 architecture:

- Custom classifier layers added to the base VGG16 model
- Specific layers are unfrozen for fine-tuning
- Uses SGD optimizer and CrossEntropyLoss
- Supports transfer learning by loading pre-trained weights

## Data Processing

- Binary files are converted to images using the `ImageGenerator` class
- Images are resized to 224x224 pixels
- Data augmentation techniques are applied (rotation, flipping, etc.)
- Dataset is split into training and validation sets

## Training Process

- Implements a shard-based training approach
- Supports resuming training from checkpoints
- Calculates and logs various metrics (accuracy, precision, recall, F1 score)
- Saves model checkpoints and training logs

## Unlearning Process

- Identifies the shards and slices containing data to be unlearned
- Retrains the affected shards while preserving other learned information
- Updates the model structure to reflect the unlearned data

