import os
import csv
import numpy as np
from typing import Any

from malwareDetector.detector import detector
from malwareDetector.config import read_config
from malwareDetector.const import DEFAULT_CONFIG_PATH

from VGG16 import VGG16
from Logger import setup_logger
from ImageGenerator import ImageGenerator

FILE_NAME = 0
LABEL = 1

class IMCFN(detector):
    def __init__(self, config_path: str=DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = read_config(config_path)
        
        self.logger = setup_logger("IMCFN", self.config.path.logging_config, self.config.path.log)
        self.logger.info('Constructing IMCFN object')
        
        self.mkdir()
        
        self.image_generator = ImageGenerator()
        self.vgg16 = VGG16(self.config.model.batch_size, self.config.model.learning_rate, self.config.model.rotation, 
                           self.config.model.width_shift, self.config.model.height_shift, self.config.model.zoom, 
                           self.config.model.shear, self.config.model.fill, self.config.model.horizontal_flip,)
            
        
    def extractFeature(self) -> None:
        '''
        Extract features from dataset.
        '''
        self.logger.info('No need to extract features')
        pass
    
    def vectorize(self) -> None:
        '''
        Vectorize dataset.
        '''
        self.logger.info('Vectorizing')
        
        labels = self.getLabel(self.config.path.input_files)
        if os.path.exists(self.config.path.test_files):
            labels.update(self.getLabel(self.config.path.test_files))
        
        labels_set = set(labels.values())
        
        # set label_idx['benignware'] = 0
        if "benignware" in labels_set:
            labels_set.remove("benignware")
            self.label_idx = {'benignware': 0}
        else:
            self.label_idx = {}
        
        self.label_idx.update({label: idx + 1 for idx, label in enumerate(labels_set)})
    
        if self.config.model.save_image == True:
            images = self.image_generator.generateImage(list(labels.keys()), self.config.folder.vectorize)
        else:
            images = self.image_generator.generateImage(list(labels.keys()))
        
        # self.images[file_name] = [image, label_idx]
        self.images = {file_name: [images[file_name][:, :, :3], label] for file_name, label in labels.items()}
        
    def model(self, training:bool=True) -> Any:
        '''
        Train model.
        '''
        self.logger.info('Setting up model')
        
        self.config = read_config(self.config_path)
        
        if self.config.path.pretrained == "DEF":
            self.vgg16.loadModel()
        elif self.config.path.pretrained is not None:
            self.vgg16.loadModel(self.config.path.pretrained)

        if training:
            selected_files = self.getFilePath(self.config.path.input_files)
            
            images = []
            labels = []
            for file_name in selected_files:
                if file_name in self.images:
                    image, label = self.images[file_name]
                    images.append(image)
                    labels.append(self.label_idx[label])
                    
            self.vgg16.loadDataFromArray(np.array(images), np.array(labels))
            self.vgg16.splitTrainData(self.config.model.train_ratio)
            train_result = self.vgg16.trainModel(self.config.model.epochs, self.config.path.log, self.config.path.model, self.config.model.print_information)
            
            if os.path.exists(self.config.path.test_files):
                test_predict = self.predict(self.config.path.test_files)
                
                test_labels = self.getLabel(self.config.path.test_files, False)
                test_labels = {file_name: label for file_name, label in test_labels.items()}
                            
                TP = TN = FP = FN = 0
                for filename in test_predict.keys():
                    true_label = test_labels[filename]
                    prediction = test_predict[filename]
                    # possitive: benignware(label = 0), negative: malware
                    if true_label == prediction:
                        if true_label == 0:
                            TP += 1
                        else:
                            TN += 1
                    elif prediction == 0:
                        FP += 1
                    else:
                        FN += 1
                        
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                
                test_result = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
                train_result['test_result'] = test_result
                
                self.logger.info(f"Testing acc: {accuracy}")
            
            return train_result
        else:
            return None
        
    def predict(self, file_path: str=None) -> dict:
        '''
        Predict the label of the file.
        '''
        self.logger.info('Predicting')
        
        if file_path is None:
            file_path = self.config.path.input_files
            
        selected_files = self.getFilePath(file_path)
        
        selected_data = [self.images[file_name][0] for file_name in selected_files]
        
        predictions = self.vgg16.predict(selected_data)
        
        selected_files = self.getFileName(file_path)
        result = {}
        for file_name, prediction in zip(selected_files, predictions):
            for label, index in self.label_idx.items():
                if index == prediction:
                    result[file_name] = label
                    break
        
        return result
    
    def getLabel(self, file_path: str, full_path: bool=True):
        '''
        Get label from csv file.
        '''
        labels = {}
        with open(file_path, 'r', newline='') as csvfile:
            rows = csv.reader(csvfile)
            next(rows) # skip header
            for row in rows:
                path = os.path.join(self.config.folder.dataset, row[FILE_NAME])
                if not os.path.exists(path):
                    self.logger.warning(f"File not found: {path}")
                    continue
                if full_path:
                    labels[path] = row[LABEL]
                else:
                    labels[row[FILE_NAME]] = row[LABEL]
        return labels
    
    def getFilePath(self, file_path: str):
        '''
        Get file path from csv file.
        '''
        with open(file_path, 'r', newline='') as csvfile:
            rows = csv.reader(csvfile)
            next(rows) # skip header
            file_path = []
            for row in rows:
                path = os.path.join(self.config.folder.dataset, row[FILE_NAME])
                if not os.path.exists(path):
                    self.logger.warning(f"File not found: {path}")
                    continue
                file_path.append(path)
        return file_path
    
    def getFileName(self, file_path: str):
        '''
        Get file name from csv file.
        '''
        with open(file_path, 'r', newline='') as csvfile:
            rows = csv.reader(csvfile)
            next(rows) # skip header
            file_name = []
            for row in rows:
                path = os.path.join(self.config.folder.dataset, row[FILE_NAME])
                if not os.path.exists(path):
                    self.logger.warning(f"File not found: {path}")
                    continue
                file_name.append(row[FILE_NAME])
        return file_name

if __name__ == '__main__':
    imcfn = IMCFN("config_origin.json")
    imcfn.vectorize()
    imcfn.model(True)
    imcfn.predict()