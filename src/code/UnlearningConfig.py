import os
import csv
import json

from malwareDetector.config import Config

DEFAULT_INPUT_CONFIG_PATH = 'dataToPython.json'

def read_unlearning_config(input_config_path=DEFAULT_INPUT_CONFIG_PATH):
    '''
    Read unlearning config from input_config_path.
    '''
    with open(input_config_path, 'r') as f:
        unlearning_config = json.load(f)
        
    for folder in unlearning_config['config']['folder'].values():
        os.makedirs(folder, exist_ok=True)
    
    # tags == 'test' -> test_files
    # else (train, predict, unlearn) -> input_files
    if unlearning_config['config']['train']:
        with open(unlearning_config['config']['path']['input_files'], 'w', newline='') as csvfile:
            with open(unlearning_config['config']['path']['test_files'], 'w', newline='') as csvfile_test:
                writer = csv.writer(csvfile)
                writer_test = csv.writer(csvfile_test)
                writer.writerow(['filename', 'label'])
                writer_test.writerow(['filename', 'label'])
                for input_data in unlearning_config['label']:
                    if input_data['tags'] == 'test':
                        writer_test.writerow([input_data['filename'], input_data['label']])
                    else:
                        writer.writerow([input_data['filename'], input_data['label']])
    else:
        with open(unlearning_config['config']['path']['input_files'], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])
            for input_data in unlearning_config['label']:
                writer.writerow([input_data['filename'], input_data['label']])
            
    return Config.parse_obj(unlearning_config['config'])
    
        