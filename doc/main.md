# main.py

## Overview

This script serves as the main entry point for the UnlearnableIMCFN system. It initializes the UnlearnableIMCFN object and executes the appropriate action (training, prediction, or unlearning) based on the configuration.

## Dependencies

- sys
- UnlearnableIMCFN (custom module)

## Main Function

### `main()`

This function is the core of the script and performs the following operations:

1. Processes command-line arguments to get the configuration path.
2. Initializes the UnlearnableIMCFN object.
3. Determines the action to perform based on the configuration.
4. Executes the appropriate method of UnlearnableIMCFN (trainModel, predict, or unlearn).

## Usage

The script can be run from the command line with an optional configuration file path:

```
python main.py [config_path]
```

If no configuration path is provided, the default configuration will be used.

## Process Flow

1. Check for a configuration file path in command-line arguments.
2. Initialize UnlearnableIMCFN with the configuration.
3. Based on the configuration:
   - If `train` is True: Call `trainModel()`
   - If `predict` is True: Call `predict()`
   - If `unlearn` is True: Call `unlearn()`

## Configuration

The behavior of the script is determined by the configuration file, which should specify one of the following actions:
- `train`: To train the model
- `predict`: To make predictions using the trained model
- `unlearn`: To perform the unlearning process

## Notes

- Only one action (train, predict, or unlearn) can be performed in a single run.
- The script modifies `sys.argv` to remove additional arguments, leaving only the script name. This may affect other parts of the code that rely on command-line arguments.
- Ensure that the UnlearnableIMCFN class and its dependencies are properly set up before running this script.

