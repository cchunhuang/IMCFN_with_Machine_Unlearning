"""
IMCFN - Image-based Malware Classification with Fine-tuning Network

This module implements the IMCFN classifier for malware detection using image-based
representation of binary files. It converts malware binaries into images and uses
a VGG16 neural network for classification.

Author: cchunhuang
Date: 2024
"""

import os
import csv
import json
import numpy as np
from typing import Any
from box import Box

from MalwareClassifier import MalwareClassifier

from VGG16 import VGG16
from Logger import setup_logger
from ImageGenerator import ImageGenerator

FILE_NAME = 0
LABEL = 1
TYPE = 2

class IMCFN(MalwareClassifier):
    def __init__(self, config_path: str="./config.json") -> None:
        """
        Initialize the IMCFN classifier.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Config Parameters Used:
            - self.config.file.logging_config: Path to logging configuration file
            - self.config.file.log: Path to log output file
            - self.config.params.model.batch_size: Batch size for training
            - self.config.params.model.learning_rate: Learning rate for optimizer
            - self.config.params.model.rotation: Rotation range for data augmentation
            - self.config.params.model.width_shift: Width shift range for data augmentation
            - self.config.params.model.height_shift: Height shift range for data augmentation
            - self.config.params.model.zoom: Zoom range for data augmentation
            - self.config.params.model.shear: Shear range for data augmentation
            - self.config.params.model.fill: Fill mode for data augmentation
            - self.config.params.model.horizontal_flip: Horizontal flip probability
        """
        super().__init__(config_path)
        
        self.logger = setup_logger("IMCFN", self.config.file.logging_config, self.config.file.log)
        self.logger.info('Constructing IMCFN object')
        
        # Initialize attributes
        self.label_idx = {}
        self.images = {}
        
        self.image_generator = ImageGenerator()
        self.vgg16 = VGG16(self.config.params.model.batch_size, self.config.params.model.learning_rate, self.config.params.model.rotation, 
                           self.config.params.model.width_shift, self.config.params.model.height_shift, self.config.params.model.zoom, 
                           self.config.params.model.shear, self.config.params.model.fill, self.config.params.model.horizontal_flip,)
            
        
    def get_feature(self) -> None:
        """
        Extract features from dataset.
        
        Note: Feature extraction is not required for IMCFN as it directly uses
        image representations of binary files.
        """
        self.logger.info('No need to extract features')
    
    def get_vector(self) -> None:
        """
        Vectorize dataset by converting binary files to images.
        
        This method reads labels from the dataset, generates image representations
        of binary files, and creates label indices for classification.
        
        Config Parameters Used:
            - self.config.file.label: Path to CSV file containing labels
            - self.config.params.vector.save_image: Whether to save generated images
            - self.config.folder.vectorize: Folder path to save images (if save_image is True)
        """
        self.logger.info('Vectorizing')
        
        labels = self.getLabel(self.config.file.label)
        
        labels_set = set(labels.values())
        
        # set label_idx['benignware'] = 0
        if "benignware" in labels_set:
            labels_set.remove("benignware")
            self.label_idx = {'benignware': 0}
        else:
            self.label_idx = {}
        
        self.label_idx.update({label: idx + 1 for idx, label in enumerate(labels_set)})
    
        if self.config.params.vector.save_image == True:
            images = self.image_generator.generateImage(list(labels.keys()), self.config.folder.vectorize)
        else:
            images = self.image_generator.generateImage(list(labels.keys()))
        
        # self.images[file_name] = [image, label_idx]
        self.images = {file_name: [images[file_name][:, :, :3], label] for file_name, label in labels.items()}
        
    def get_model(self, action: str="train") -> Any:
        """
        Train model or perform inference.
        
        Args:
            action: Operation mode - "train" for training, "predict" for inference
            
        Returns:
            Training results dictionary if action is "train", None otherwise.
            The dictionary contains metrics like accuracy, precision, recall, F1-score,
            and training history.
            
        Config Parameters Used:
            - self.config.file.pretrained: Path to pretrained model or "DEF" for default
            - self.config.file.label: Path to CSV file containing labels
            - self.config.params.model.train_ratio: Ratio of training data to validation data
            - self.config.params.model.epochs: Number of training epochs
            - self.config.file.log: Path to training log file
            - self.config.file.model: Path to save trained model
            - self.config.params.model.print_information: Additional information to log
        """
        self.logger.info('Setting up model')
        
        # Reload config
        with open(self.config_path, encoding='utf-8') as f:
            self.config = Box(json.load(f))
        
        training = (action == "train")
        
        if self.config.file.pretrained == "DEF":
            self.vgg16.loadModel()
        elif self.config.file.pretrained is not None:
            self.vgg16.loadModel(self.config.file.pretrained)

        if training:
            selected_files = self.getFilePath(self.config.file.label, 'train')
            
            images = []
            labels = []
            for file_name in selected_files:
                if file_name in self.images:
                    image, label = self.images[file_name]
                    images.append(image)
                    labels.append(self.label_idx[label])
                    
            self.vgg16.loadDataFromArray(np.array(images), np.array(labels))
            self.vgg16.splitTrainData(self.config.params.model.train_ratio)
            train_result = self.vgg16.trainModel(self.config.params.model.epochs, self.config.file.log, self.config.file.model, self.config.params.model.print_information)
            
            # Check if there are test data in the label file
            test_labels = self.getLabel(self.config.file.label, full_path=False, data_type='test')
            if len(test_labels) > 0:
                test_predict = self.predict(data_type='test')
                
                            
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
                
                self.logger.info("Testing acc: %.4f", accuracy)
            
            return train_result
        else:
            return None
        
    def get_prediction(self, data_type: str=None) -> dict:
        """
        Predict the label of files.
        
        Args:
            data_type: Type of data to predict ('train', 'test', or None for all)
            
        Returns:
            Dictionary mapping file names to predicted labels
            
        Config Parameters Used:
            - self.config.file.label: Path to CSV file containing labels
        """
        self.logger.info('Predicting')
        
        selected_files = self.getFilePath(self.config.file.label, data_type)
        
        # Filter out files that are not in self.images
        valid_files = [file_name for file_name in selected_files if file_name in self.images]
        selected_data = [self.images[file_name][0] for file_name in valid_files]
        
        predictions = self.vgg16.predict(selected_data)
        
        selected_files = self.getFileName(self.config.file.label, data_type)
        result = {}
        for file_name, prediction in zip(selected_files, predictions):
            for label, index in self.label_idx.items():
                if index == prediction:
                    result[file_name] = label
                    break
        
        return result
    
    def getLabel(self, file_path: str, full_path: bool=True, data_type: str=None):
        """
        Get label from CSV file.
        
        Args:
            file_path: Path to CSV file containing labels
            full_path: If True, return full path; if False, return file name only
            data_type: Filter by type column ('train', 'test', or None for all)
            
        Returns:
            Dictionary mapping file paths/names to their labels
            
        Config Parameters Used:
            - self.config.folder.dataset: Base folder path for dataset files
        """
        labels = {}
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            rows = csv.reader(csvfile)
            next(rows) # skip header
            for row in rows:
                # Skip if data_type filter is specified and doesn't match
                if data_type is not None and len(row) > TYPE and row[TYPE] != data_type:
                    continue
                    
                path = os.path.join(self.config.folder.dataset, row[FILE_NAME])
                if not os.path.exists(path):
                    self.logger.warning("File not found: %s", path)
                    continue
                if full_path:
                    labels[path] = row[LABEL]
                else:
                    labels[row[FILE_NAME]] = row[LABEL]
        return labels
    
    def getFilePath(self, file_path: str, data_type: str=None):
        """
        Get file paths from CSV file.
        
        Args:
            file_path: Path to CSV file
            data_type: Filter by type column ('train', 'test', or None for all)
            
        Returns:
            List of file paths
            
        Config Parameters Used:
            - self.config.folder.dataset: Base folder path for dataset files
        """
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            rows = csv.reader(csvfile)
            next(rows) # skip header
            file_paths = []
            for row in rows:
                # Skip if data_type filter is specified and doesn't match
                if data_type is not None and len(row) > TYPE and row[TYPE] != data_type:
                    continue
                    
                path = os.path.join(self.config.folder.dataset, row[FILE_NAME])
                if not os.path.exists(path):
                    self.logger.warning("File not found: %s", path)
                    continue
                file_paths.append(path)
        return file_paths
    
    def getFileName(self, file_path: str, data_type: str=None):
        """
        Get file names from CSV file.
        
        Args:
            file_path: Path to CSV file
            data_type: Filter by type column ('train', 'test', or None for all)
            
        Returns:
            List of file names
            
        Config Parameters Used:
            - self.config.folder.dataset: Base folder path for dataset files
        """
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            rows = csv.reader(csvfile)
            next(rows) # skip header
            file_names = []
            for row in rows:
                # Skip if data_type filter is specified and doesn't match
                if data_type is not None and len(row) > TYPE and row[TYPE] != data_type:
                    continue
                    
                path = os.path.join(self.config.folder.dataset, row[FILE_NAME])
                if not os.path.exists(path):
                    self.logger.warning("File not found: %s", path)
                    continue
                file_names.append(row[FILE_NAME])
        return file_names

if __name__ == '__main__':
    imcfn = IMCFN("config_origin.json")
    imcfn.get_vector()
    imcfn.get_model("train")
    imcfn.get_prediction()