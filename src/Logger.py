"""
Logger Configuration Module

This module provides logging configuration utilities for the IMCFN project.
It integrates with MalwareClassifier's logging system while allowing project-specific
customization through configuration files.

Author: cchunhuang
Date: 2024
"""

import os
import json
import logging
from MalwareClassifier import setup_logging, get_logger

def setup_logger(logger_name: str, logging_config_path: str="", output_log_path: str=None):
    """
    Setup logger using MalwareClassifier's logging system with project-specific configuration.
    
    This function uses MalwareClassifier's logging infrastructure but maintains control
    over logging configuration through this project's logging_config.json file.
    
    Args:
        logger_name: Name of the logger
        logging_config_path: Path to project's logging config file (e.g., logging_config.json)
        output_log_path: Path to output log file (overrides config file setting)
    
    Returns:
        Logger instance
    """
    # Load project-specific logging configuration if provided
    custom_config = None
    if logging_config_path and os.path.exists(logging_config_path):
        with open(logging_config_path, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        
        # Override the log file path if output_log_path is specified
        if output_log_path and 'handlers' in custom_config and 'file' in custom_config['handlers']:
            custom_config['handlers']['file']['filename'] = output_log_path
    else:
        # If config file doesn't exist or not specified, setup basic logging
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.propagate = True
        return logger
    
    # Determine log directory
    log_dir = None
    if output_log_path:
        log_dir = os.path.dirname(output_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    elif custom_config and 'handlers' in custom_config and 'file' in custom_config['handlers']:
        # Extract log directory from custom config
        log_file = custom_config['handlers']['file'].get('filename', '')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with MalwareClassifier, passing custom config if available
    setup_logging(config=custom_config, log_dir=log_dir)
    
    # Get and return the logger
    return get_logger(logger_name)