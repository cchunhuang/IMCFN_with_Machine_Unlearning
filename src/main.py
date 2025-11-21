"""
Main Entry Point for UnlearnableIMCFN

This module serves as the main entry point for running the UnlearnableIMCFN system.
It handles command-line arguments and dispatches to appropriate actions (train, predict).

Usage:
    python main.py [config_path]

Author: IMCFN Project Team
Date: 2024
"""

import sys

from UnlearnableIMCFN import UnlearnableIMCFN

def main():
    """
    Main function of UnlearnableIMCFN.
    
    Reads configuration file path from command line arguments (optional),
    initializes the UnlearnableIMCFN system, and executes the specified action.
    """
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = None
    
    sys.argv = [sys.argv[0]]
    unlearnable_imcfn = UnlearnableIMCFN(config_path)
    
    if unlearnable_imcfn.config.action == "train":
        unlearnable_imcfn.trainModel()
    elif unlearnable_imcfn.config.action == "predict":
        unlearnable_imcfn.predict()
    elif unlearnable_imcfn.config.action == "unlearn":
        unlearnable_imcfn.unlearn()
    else:
        unlearnable_imcfn.logger.error("Unknown action: %s", unlearnable_imcfn.config.action)
        
if __name__ == '__main__':
    main()