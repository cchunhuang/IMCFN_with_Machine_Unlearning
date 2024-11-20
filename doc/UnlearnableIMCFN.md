# UnlearnableIMCFN.py

## Overview

This script defines the `UnlearnableIMCFN` class, which implements a machine learning model for malware detection with unlearning capabilities. It uses the IMCFN (Image-based Malware Classification with Feature Networks) approach and incorporates a shard-based training mechanism for efficient unlearning.

## Dependencies

- os
- csv
- json
- time
- copy
- random
- malwareDetector.config
- UnlearningConfig (custom module)
- IMCFN (custom module)
- Logger (custom module)

## Class: UnlearnableIMCFN

### Constructor

```python
def __init__(self, config_path: str = None)
```

Initializes the UnlearnableIMCFN object with the specified configuration path. This method performs the following key operations:
- Loads the configuration
- Sets up paths and folders
- Initializes the logger
- If training mode is enabled:
  - Splits the dataset into shards and slices based on the configuration
  - Saves the position information of data in shards and slices
- Initializes the IMCFN object with a customized configuration

### Methods

#### `trainModel(self, shard_list: list = None, start_slice: list = None)`

Trains the model using the specified shards and starting slices.

Parameters:
- `shard_list: list = None`: Specifies which shards to train.
  - If None, all shards are trained.
  - If provided, it should be a list of integers representing the indices of shards to train.
  - Example: `[0, 2, 3]` will train only the shards with indices 0, 2, and 3.

- `start_slice: list = None`: Specifies the starting slice for each shard.
  - If None, training starts from the first slice of each shard.
  - If provided, it should be a list of integers with length equal to the total number of shards, where each integer represents the starting slice index for the corresponding shard.
  - Example: For 4 shards, `[0, 2, 1, 3]` means start training from slice 0 for the first shard, slice 2 for the second shard, and so on.

These parameters allow for:
- Selective training of specific shards instead of all shards every time.
- Starting training from specific slices in each shard, useful for incremental training or resuming interrupted training.

Example usage:
```python
uv.trainModel(shard_list=[0, 2], start_slice=[2, 0, 0, 0])
```
This will train shards 0 and 2, starting from slice 2 in shard 0 and slice 0 in shard 2. Note that `start_slice` still needs a value for all shards, even if not all are being trained.

#### `predict(self)`

Predicts labels for the input files using the trained model.
- Uses the latest trained model from each shard
- Implements a voting mechanism across subdetectors
- Saves the prediction results to a JSON file

#### `unlearn(self)`

Removes specified data from the model and retrains affected shards.
- Identifies the shards and slices containing the data to be unlearned
- Removes the specified data
- Retrains the affected shards and slices

#### `savePosition(self)`

Saves the current position of data in shards and slices.

#### `getPosition(self)`

Retrieves the position of data in shards and slices.

#### `saveSubdetectorLabel(self, data: list)`

Saves labels for the subdetector.

#### `saveSubdetectorName(self, model_name: list)`

Saves names of subdetector models.

#### `getSubdetectorName(self)`

Retrieves names of subdetector models.

#### `getLabel(self)`

Retrieves labels for the input files.

#### `getFileName(self)`

Retrieves filenames of the input files.

## Key Features

1. **Shard-based Training**: Divides data into shards and slices during initialization for efficient training and unlearning.
2. **Unlearning Capability**: Allows removal of specific data points from the trained model.
3. **Voting-based Prediction**: Uses multiple subdetectors for prediction, employing a voting mechanism.
4. **Flexible Configuration**: Uses a JSON configuration file for setup.
5. **Logging**: Implements comprehensive logging throughout the process.
6. **Performance Metrics**: Calculates and saves various performance metrics including accuracy, precision, recall, and F1 score.

## Process Flow

1. Initialize UnlearnableIMCFN object with configuration (including data splitting into shards and slices)
2. Train the model using shards and slices
3. Predict labels for new samples
4. Unlearn specific data points if needed

## Usage Example

```python
config_path = './output/config/dataToPython.json'

# Training
uv = UnlearnableIMCFN(config_path)
uv.trainModel()

# Prediction
predict = uv.predict()

# Unlearning
uv.unlearn()
```

## Notes

- The class uses a configuration object to store various settings and paths.
- During initialization, the dataset is divided into shards and slices for efficient processing and unlearning.
- The `unlearn` method removes specified data and retrains affected shards.
- Prediction uses a voting mechanism across multiple subdetectors.
- Performance metrics and model details are saved in JSON format.
- The class relies on custom modules (IMCFN, Logger, UnlearningConfig) which should be properly implemented and accessible.
- Ensure that the configuration file is properly set up before using this class.
- Different configuration files may be used for training, prediction, and unlearning processes.

