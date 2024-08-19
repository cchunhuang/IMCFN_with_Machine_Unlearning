# Logger.py

## Overview

This script provides a utility function `setup_logger` for configuring and setting up a logger in Python. It allows for flexible logger configuration using either a JSON configuration file or default settings.

## Dependencies

- os
- json
- logging
- logging.config

## Function: setup_logger

### Signature

```python
def setup_logger(logger_name: str, logging_config_path: str="", output_log_path: str=None) -> logging.Logger
```

### Parameters

- `logger_name` (str): The name of the logger to be created or retrieved.
- `logging_config_path` (str, optional): Path to the JSON logging configuration file. If not provided or the file doesn't exist, default settings will be used.
- `output_log_path` (str, optional): Path where log files should be written. If provided, this will override the log file path in the JSON configuration.

### Returns

- `logging.Logger`: A configured logger object.

### Functionality

1. Retrieves or creates a logger with the specified `logger_name`.
2. If `logging_config_path` is not provided or the file doesn't exist:
   - Sets the logger level to INFO.
   - Clears any existing handlers.
   - Enables propagation of log messages to parent loggers.
3. If `logging_config_path` is provided and the file exists:
   - Loads the logging configuration from the JSON file.
   - If `output_log_path` is provided, updates the file handler's filename in the configuration.
   - Applies the configuration using `logging.config.dictConfig()`.
4. Returns the configured logger.

## Usage Example

```python
from Logger import setup_logger

# Basic usage with default settings
logger = setup_logger("MyLogger")

# Usage with a configuration file
logger = setup_logger("MyLogger", "path/to/logging_config.json")

# Usage with a configuration file and custom output path
logger = setup_logger("MyLogger", "path/to/logging_config.json", "path/to/output.log")

# Using the logger
logger.info("This is an info message")
logger.error("This is an error message")
```

## Notes

- If a logging configuration file is not provided or cannot be found, the function will set up a basic logger with INFO level logging.
- The JSON configuration file should follow the structure expected by `logging.config.dictConfig()`.
- When providing a custom output log path, ensure that the directory exists and is writable.
- This setup allows for centralized logging configuration across multiple modules in a project.

