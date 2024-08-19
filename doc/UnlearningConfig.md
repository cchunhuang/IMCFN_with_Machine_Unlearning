# UnlearningConfig.py

## Overview

This script provides functionality to read and process configuration data for an unlearning system, likely related to a malware detection project. It reads a JSON configuration file, creates necessary directories, and generates CSV files for input and test data.

## Dependencies

- os
- csv
- json
- Config (from malwareDetector.config)

## Constants

- `DEFAULT_INPUT_CONFIG_PATH`: Default path for the input configuration file, set to 'dataToPython.json'.

## Function: read_unlearning_config

### Signature

```python
def read_unlearning_config(input_config_path=DEFAULT_INPUT_CONFIG_PATH)
```

### Parameters

- `input_config_path` (str, optional): Path to the input configuration JSON file. Defaults to `DEFAULT_INPUT_CONFIG_PATH`.

### Returns

- `Config`: A Config object parsed from the configuration data.

### Functionality

1. Reads the JSON configuration file specified by `input_config_path`.
2. Creates directories specified in the configuration's 'folder' section.
3. Generates two CSV files:
   - An input file for training, prediction, and unlearning data.
   - A test file for test data.
4. Populates these CSV files with filename and label data from the configuration.
5. Returns a `Config` object containing the parsed configuration data.

## Process Flow

1. Open and load the JSON configuration file.
2. Create all directories specified in the configuration's 'folder' section.
3. Open two CSV files for writing: one for input data and one for test data.
4. Write headers to both CSV files: 'filename' and 'label'.
5. Iterate through the 'label' list in the configuration:
   - If an item's 'tags' is 'test', write it to the test CSV file.
   - Otherwise, write it to the input CSV file.
6. Parse the configuration data into a `Config` object and return it.

## Usage Example

```python
from UnlearningConfig import read_unlearning_config

# Using default configuration path
config = read_unlearning_config()

# Using a custom configuration path
custom_config = read_unlearning_config('path/to/custom_config.json')

# Accessing configuration data
input_folder = config.folder.input
output_folder = config.folder.output
```

## Notes

- The configuration file should be a JSON file with a specific structure, including 'config' and 'label' sections.
- The 'config' section should include 'folder' and 'path' subsections.
- The 'label' section should be a list of dictionaries, each containing 'filename', 'label', and 'tags' keys.
- The script creates two types of CSV files: one for general input (training, prediction, unlearning) and another specifically for test data.
- Ensure that the JSON configuration file is properly formatted to avoid runtime errors.
- The `Config` class used in this script is imported from `malwareDetector.config` and should be compatible with the structure of the parsed configuration data.

