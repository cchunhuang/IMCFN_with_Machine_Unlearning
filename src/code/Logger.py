import os
import json
import logging
import logging.config

def setup_logger(logger_name: str, logging_config_path: str="", output_log_path: str=None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    
    if not os.path.exists(logging_config_path): 
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.propagate = True
    else:
        with open(logging_config_path, 'r') as f:
            logging_config = json.load(f)
            if output_log_path is not None:
                logging_config['handlers']['file']['filename'] = output_log_path
        logging.config.dictConfig(logging_config)
        
    return logger