"""
UnlearnableIMCFN - IMCFN with Machine Unlearning Capabilities

This module implements an extended version of IMCFN that supports machine unlearning.
It uses a sharded and sliced training approach to enable efficient removal of specific
training data without complete retraining.

Author: cchunhuang
Date: 2024
"""

import os
import csv
import json
import time
import copy
import random
from box import Box

from IMCFN import IMCFN
from Logger import setup_logger

FILE_NAME = 0
LABEL = 1
TYPE = 2

DEFAULT_CONFIG_PATH = './config.json'

class UnlearnableIMCFN:
    def __init__(self, config_path: str=None):
        """
        Initialize the UnlearnableIMCFN classifier.
        
        This constructor sets up the sharded training structure and initializes
        the underlying IMCFN model for each shard.
        
        Args:
            config_path: Path to configuration JSON file (default: './config.json')
            
        Config Parameters Used:
            - self.config.folder: All folder paths (model, dataset, etc.)
            - self.config.params.model.shard: Number of shards for training
            - self.config.params.model.slice: Number of slices per shard
            - self.config.params.model.batch_size: Batch size for data distribution
            - self.config.params.model.overwrite: Whether to overwrite existing files
            - self.config.file.logging_config: Path to logging configuration
            - self.config.file.log: Path to log output file
            - self.config.action: Action to perform ('train', 'predict', or 'unlearn')
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
            
        # Read config from JSON file
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create folders
        for folder in config_data['folder'].values():
            os.makedirs(folder, exist_ok=True)
        
        # Convert to Box for easy access
        self.config = Box(config_data)
        
        # -----set paths-----
        
        self.model_folder = os.path.join(self.config.folder.model, time.strftime("%Y%m%d_%H%M", time.localtime()))
        
        self.subdetector_shard_folder = [os.path.join(self.model_folder, f"shard{i}") for i in range(self.config.params.model.shard)]
        for shard_folder in self.subdetector_shard_folder:
            os.makedirs(shard_folder, exist_ok=True)
            
        self.subdetector_label_path = os.path.join(self.model_folder, "subdetector_label.csv")
        self.subdetector_config_path = os.path.join(self.model_folder, "subdetector_config.json")
        
        # -----load logger-----
        
        self.config.file.log = os.path.join(self.model_folder, 'loging.log')
        self.logger = setup_logger("UnlearnableIMCFN", self.config.file.logging_config, self.config.file.log)
        self.logger.info('Constructing UnlearnableIMCFN object')
        
        # -----spilt data into shards and slices-----
        
        if self.config.action == "train":
            if self.config.params.model.overwrite is False:
                self.config.file.position = os.path.join(self.model_folder, 'position.csv')
                self.config.file.subdetector_name = os.path.join(self.model_folder, 'subdetector_name.csv')
            
            labels = self.getLabel(data_type='train')
                
            SEED_VALUE = 42
            random.seed(SEED_VALUE)
            random.shuffle(labels)
                
            self.input_data = [[[] for j in range(self.config.params.model.slice)] for i in range(self.config.params.model.shard)]
            for i in range(0, len(labels), self.config.params.model.batch_size):
                shard_idx = i // self.config.params.model.batch_size % self.config.params.model.shard
                slice_idx = i // self.config.params.model.batch_size // self.config.params.model.shard % self.config.params.model.slice
                for j in range(i, min(i + self.config.params.model.batch_size, len(labels))):
                    self.input_data[shard_idx][slice_idx].append(labels[j])
            
            self.savePosition()
        
        #-----initialize IMCFN-----
        
        subdetector_config = copy.deepcopy(self.config)
        subdetector_config.file.pop('position', None)
        subdetector_config.file.pop('subdetector_name', None)
        subdetector_config.params.model.pop('shard', None)
        subdetector_config.params.model.pop('slice', None)
        
        subdetector_config.file.log = self.config.file.log
        
        with open(self.subdetector_config_path, 'w', encoding='utf-8') as f:
            json.dump(subdetector_config.to_dict(), f, indent=4)
        
        self.imcfn = IMCFN(self.subdetector_config_path)
        
    def trainModel(self, shard_list: list=None, start_slice: list=None):
        """
        Train the model using sharded and sliced approach.
        
        Args:
            shard_list: List of shard indices to train (default: all shards)
            start_slice: List of starting slice indices for each shard (default: 0 for all)
            
        Note:
            The model is trained incrementally on slices within each shard,
            enabling efficient unlearning later.
            
        Config Parameters Used:
            - self.config.params.model.shard: Number of shards
            - self.config.params.model.slice: Number of slices per shard
            - self.config.params.model.overwrite: Whether to overwrite existing files
            - self.config.params.model.epochs: Number of training epochs
            - self.config.file.label: Path to label file
            - self.config.file.subdetector_name: Path to subdetector name file
            - self.config.file.score: Path to save training scores
        """
        self.logger.info('Training model')
        
        # -----Validation-----
        
        if shard_list is None:
            shard_list = [i for i in range(self.config.params.model.shard)]
        else:
            if not 1 <= len(shard_list) <= self.config.params.model.shard:
                raise ValueError(f"shard_list size must be between 1 and {self.config.params.model.shard}")
            for item in shard_list:
                if not isinstance(item, int) or not 0 <= item < self.config.params.model.shard:
                    raise ValueError(f"All items in shard_list must be integers between 0 and {self.config.params.model.shard - 1}")
            
        if start_slice is None:
            start_slice = [0 for i in range(self.config.params.model.shard)]
        else:
            if len(start_slice) != self.config.params.model.shard:
                raise ValueError(f"start_slice size must be exactly {self.config.params.model.shard}")
            for item in start_slice:
                if not isinstance(item, int) or not 0 <= item < self.config.params.model.slice:
                    raise ValueError(f"All items in start_slice must be integers between 0 and {self.config.params.model.slice - 1}")
                
        # -----vectorize-----        
        
        self.imcfn.get_vector()
                
        # -----initialize subdetector name-----
        
        if not os.path.exists(self.config.file.subdetector_name):
            subdetector_name = [["DEF"] + [None for j in range(self.config.params.model.slice)] for i in range(self.config.params.model.shard)]
        else:                
            subdetector_name = self.getSubdetectorName()   
            
        # -----process config-----
        
        with open(self.subdetector_config_path, encoding='utf-8') as f:
            subdetector_config = Box(json.load(f))
        subdetector_config.file.label = self.subdetector_label_path
        
        # -----train model-----
        
        score = []
        
        for shard_idx in shard_list:
            # load pretrained model
            subdetector_config.file.pretrained = None if subdetector_name[shard_idx][start_slice[shard_idx]] == '' else subdetector_name[shard_idx][start_slice[shard_idx]]
            with open(self.subdetector_config_path, 'w', encoding='utf-8') as f:
                json.dump(subdetector_config.to_dict(), f, indent=4)
            
            self.imcfn.get_model("predict")
            
            subdetector_config.file.pretrained = None
            
            # get data
            data = [_data for _slice_idx in range(start_slice[shard_idx]) for _data in self.input_data[shard_idx][_slice_idx]]
            
            for slice_idx in range(start_slice[shard_idx], self.config.params.model.slice):
                data.extend(self.input_data[shard_idx][slice_idx])
                self.saveSubdetectorLabel(data)
                
                subdetector_name[shard_idx][slice_idx + 1] = os.path.join(self.subdetector_shard_folder[shard_idx], f"slice{slice_idx}.pth")
                
                subdetector_config.file.log = os.path.join(self.subdetector_shard_folder[shard_idx], "log.txt")
                subdetector_config.file.model = subdetector_name[shard_idx][slice_idx + 1]
                subdetector_config.params.model.print_information = f"*****Shard: {shard_idx} | Slice: {slice_idx}*****"
                with open(self.subdetector_config_path, 'w', encoding='utf-8') as f:
                    json.dump(subdetector_config.to_dict(), f, indent=4)
                
                result = self.imcfn.get_model("train")
                
                train_acc = [{"epoch": i, "accuracy": result['train_acc'][i]} for i in range(len(result['train_acc']))]
                train_loss = [{"epoch": i, "loss": result['train_loss'][i]} for i in range(len(result['train_loss']))]
                val_acc = [{"epoch": i, "accuracy": result['valid_acc'][i]} for i in range(len(result['valid_acc']))]
                val_loss = [{"epoch": i, "loss": result['valid_loss'][i]} for i in range(len(result['valid_loss']))]
                test_acc = result.get('test_result', {}).get('accuracy', None)
                
                score.append({"model_name": subdetector_name[shard_idx][slice_idx + 1], "train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc})
        
        # -----save model-----
        
        self.saveSubdetectorName(subdetector_name)
        
        # -----test model-----
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        # Check if there are test data
        test_labels = self.getLabel(data_type='test')
        if len(test_labels) > 0:
            subdetector_config.file.label = self.config.file.label
            with open(self.subdetector_config_path, 'w', encoding='utf-8') as f:
                json.dump(subdetector_config.to_dict(), f, indent=4)
            
            # Predict on test dataset using voting from all shards
            test_file_names = [label[FILE_NAME] for label in test_labels]
            test_labels_dict = {test_labels[i][FILE_NAME]: test_labels[i][LABEL] for i in range(len(test_labels))}
            
            vote = {file_name: {} for file_name in test_file_names}
            
            LATEST_MODEL = -1
            
            for shard_idx in range(self.config.params.model.shard):
                if subdetector_name[shard_idx][LATEST_MODEL] == None or subdetector_name[shard_idx][LATEST_MODEL] == '':
                    self.logger.warning("Subdetector shard%s is not trained, skipping test", shard_idx)
                    continue
                
                subdetector_config.file.pretrained = subdetector_name[shard_idx][LATEST_MODEL]
                with open(self.subdetector_config_path, 'w', encoding='utf-8') as f:
                    json.dump(subdetector_config.to_dict(), f, indent=4)
                
                self.imcfn.get_model("predict")
                predict_result = self.imcfn.get_prediction(data_type='test')
                for file_name in predict_result.keys():
                    if predict_result[file_name] not in vote[file_name].keys():
                        vote[file_name][predict_result[file_name]] = 1
                    else:
                        vote[file_name][predict_result[file_name]] += 1
            
            # Calculate metrics
            for file_name in vote.keys():
                if len(vote[file_name]) == 0:
                    continue
                true_label = test_labels_dict[file_name]
                prediction = max(vote[file_name], key=vote[file_name].get)
                # possitive: benignware(label = 0), negative: malware
                if true_label == prediction:
                    if true_label == 'benignware':
                        TP += 1
                    else:
                        TN += 1
                elif prediction == 'benignware':
                    FP += 1
                else:
                    FN += 1
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        final_result = {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
        
        # Log test results
        if len(test_labels) > 0:
            self.logger.info("=" * 50)
            self.logger.info("Test Results:")
            self.logger.info("TP: %d, TN: %d, FP: %d, FN: %d", TP, TN, FP, FN)
            self.logger.info("Accuracy: %.4f", accuracy)
            self.logger.info("Precision: %.4f", precision)
            self.logger.info("Recall: %.4f", recall)
            self.logger.info("F1-Score: %.4f", f1_score)
            self.logger.info("=" * 50)
        else:
            self.logger.info("No test data available for evaluation")
        
        #-----save score-----
        
        if self.config.params.model.overwrite is False:
            self.config.file.score = os.path.join(self.model_folder, "score.json")
        
        with open(self.config.file.score, 'w', encoding='utf-8') as f:
            json.dump([{"final_result": final_result}] + score, f, indent=4)
        
    def predict(self):
        """
        Predict labels using ensemble voting across all shards.
        
        Returns:
            List of dictionaries containing file names and predicted labels
            
        Config Parameters Used:
            - self.config.params.model.shard: Number of shards
            - self.config.params.model.overwrite: Whether to overwrite existing results
            - self.config.file.result: Path to save prediction results
        """
        self.logger.info('Predicting')
        
        self.imcfn.get_vector()
        
        file_names = self.getFileName()
        subdetector_name = self.getSubdetectorName()
        with open(self.subdetector_config_path, encoding='utf-8') as f:
            subdetector_config = Box(json.load(f))
        
        vote = {file_name: {} for file_name in file_names}
        
        LATEST_MODEL = -1
        
        for shard_idx in range(self.config.params.model.shard):
            if subdetector_name[shard_idx][LATEST_MODEL] == None or subdetector_name[shard_idx][LATEST_MODEL] == '':
                raise ValueError(f"Subdetector shard{shard_idx} is not trained")
            
            subdetector_config.file.pretrained = subdetector_name[shard_idx][LATEST_MODEL]
            with open(self.subdetector_config_path, 'w', encoding='utf-8') as f:
                json.dump(subdetector_config.to_dict(), f, indent=4)
            
            self.imcfn.get_model("predict")
            predict_result = self.imcfn.get_prediction()
            for file_name in predict_result.keys():
                if predict_result[file_name] not in vote[file_name].keys():
                    vote[file_name][predict_result[file_name]] = 1
                else:
                    vote[file_name][predict_result[file_name]] += 1
        
        result = [{"name": file_name, "detection": max(vote[file_name], key=vote[file_name].get)} for file_name in predict_result.keys()]
        
        if self.config.params.model.overwrite is False:
            self.config.file.result = os.path.join(self.model_folder, "predict_result.json")
        
        with open(self.config.file.result, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
            
        return result
    
    def unlearn(self):
        """
        Remove specific training samples from the model (machine unlearning).
        
        This method identifies which shards and slices contain the data to be removed,
        and retrains only those affected portions of the model.
        
        Config Parameters Used:
            - self.config.params.model.shard: Number of shards
            - self.config.params.model.slice: Number of slices per shard
            - self.config.params.model.overwrite: Whether to overwrite existing files
            - self.config.file.position: Path to position tracking file
            - self.config.file.subdetector_name: Path to subdetector name file
            - self.config.file.label: Path to label file
        """
        self.logger.info('Unlearning')
        
        self.getPosition()
        file_names = self.getFileName()
        
        shard_list = set()
        start_slice = [self.config.params.model.slice - 1 for i in range(self.config.params.model.shard)]
        
        for unlearn_data in file_names:
            flag = False
            for shard_idx in range(self.config.params.model.shard):
                for slice_idx in range(self.config.params.model.slice):
                    for data in self.input_data[shard_idx][slice_idx]:
                        if data[FILE_NAME] == unlearn_data:
                            self.input_data[shard_idx][slice_idx].remove(data)
                            
                            shard_list.add(shard_idx)
                            start_slice[shard_idx] = min(slice_idx, start_slice[shard_idx])
                            
                            flag = True
                            break
                if flag:
                    break
            if not flag:
                self.logger.warning('The file was not learned: %s', unlearn_data)
        
        if self.config.params.model.overwrite is False:
            self.config.file.position = os.path.join(self.model_folder, 'position.csv')
            self.config.file.subdetector_name = os.path.join(self.model_folder, 'subdetector_name.csv')
            
        self.savePosition()
        
        shard_list = list(shard_list)
        if len(shard_list) != 0:
            self.logger.info('Retrain shard: %s', shard_list)
            self.logger.info('Start slice: %s', start_slice)
            
            with open(self.config.file.label, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['file_name', 'label', 'type'])
                for shard_idx in shard_list:
                    for slice_idx in range(self.config.params.model.slice):
                        for data in self.input_data[shard_idx][slice_idx]:
                            writer.writerow([data[FILE_NAME], data[LABEL], 'train'])
                    
            self.trainModel(shard_list, start_slice)
                
    def savePosition(self):
        """
        Save the position (shard and slice) of each data sample to a CSV file.
        
        This information is crucial for efficient machine unlearning.
        
        Config Parameters Used:
            - self.config.file.position: Path to save position tracking file
        """
        with open(self.config.file.position, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'label', 'shard', 'slice'])
            for shard_idx in range(len(self.input_data)):
                for slice_idx in range(len(self.input_data[shard_idx])):
                    for data in self.input_data[shard_idx][slice_idx]:
                        writer.writerow([data[FILE_NAME], data[LABEL], shard_idx, slice_idx])
        
    def getPosition(self):
        """
        Load the position (shard and slice) of each data sample from CSV file.
        
        Raises:
            FileNotFoundError: If position file does not exist
            
        Config Parameters Used:
            - self.config.file.position: Path to position tracking file
            - self.config.params.model.slice: Number of slices per shard
            - self.config.params.model.shard: Number of shards
        """
        if not os.path.exists(self.config.file.position):
            raise FileNotFoundError(f"File not found: {self.config.file.position}")
        
        SHARD_IDX = 2
        SLICE_IDX = 3
        
        self.input_data = [[[] for j in range(self.config.params.model.slice)] for i in range(self.config.params.model.shard)]
        with open(self.config.file.position, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for rows in reader:
                self.input_data[int(rows[SHARD_IDX])][int(rows[SLICE_IDX])].append([rows[FILE_NAME], rows[LABEL]])
        
    def saveSubdetectorLabel(self, data: list):
        """
        Save labels for the current subdetector training.
        
        Args:
            data: List of data samples with file names and labels
        """
        with open(self.subdetector_label_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'label', 'type'])
            for i in range(len(data)):
                writer.writerow([data[i][FILE_NAME], data[i][LABEL], 'train'])
        
    def saveSubdetectorName(self, model_name: list):
        """
        Save the model file paths for all subdetectors.
        
        Args:
            model_name: 2D list of model paths organized by shard and slice
            
        Config Parameters Used:
            - self.config.file.subdetector_name: Path to subdetector name file
            - self.config.params.model.slice: Number of slices per shard
        """
        with open(self.config.file.subdetector_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['default'] + [f'slice{i}' for i in range(self.config.params.model.slice)])
            for i in range(self.config.params.model.shard):
                writer.writerow(model_name[i])       
                
    def getSubdetectorName(self):
        """
        Load the model file paths for all subdetectors.
        
        Returns:
            2D list of model paths organized by shard and slice
            
        Config Parameters Used:
            - self.config.file.subdetector_name: Path to subdetector name file
        """
        with open(self.config.file.subdetector_name, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            return [rows for rows in reader]  
        
    def getLabel(self, path: str=None, data_type: str=None):
        """
        Get labels from CSV file.
        
        Args:
            path: Path to CSV file (default: config.file.label)
            data_type: Filter by type column ('train', 'test', or None for all)
            
        Returns:
            List of [file_name, label] pairs
            
        Config Parameters Used:
            - self.config.file.label: Default path to label file (if path is None)
            - self.config.folder.dataset: Base folder path for dataset files
        """
        if path is None:
            path = self.config.file.label
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            labels = []
            for rows in reader:
                # Skip if data_type filter is specified and doesn't match
                if data_type is not None and len(rows) > TYPE and rows[TYPE] != data_type:
                    continue
                    
                if not os.path.exists(os.path.join(self.config.folder.dataset, rows[FILE_NAME])):
                    self.logger.warning("File not found: %s", os.path.join(self.config.folder.dataset, rows[FILE_NAME]))
                    continue
                labels.append([rows[FILE_NAME], rows[LABEL]])
        return labels
    
    def getFileName(self, data_type: str=None):
        """
        Get file names from CSV file.
        
        Args:
            data_type: Filter by type column ('train', 'test', or None for all)
            
        Returns:
            List of file names
            
        Config Parameters Used:
            - self.config.file.label: Path to label file
        """
        with open(self.config.file.label, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            file_names = []
            for rows in reader:
                # Skip if data_type filter is specified and doesn't match
                if data_type is not None and len(rows) > TYPE and rows[TYPE] != data_type:
                    continue
                file_names.append(rows[FILE_NAME])
        return file_names
        
if __name__ == '__main__':
    # train
    train_config_path = './output/config/dataToPython.json'
    uv = UnlearnableIMCFN(train_config_path)
    uv.trainModel()
    
    #predict
    # predict_config_path = './output/config/dataToPython_predict.json'
    # uv = UnlearnableIMCFN(predict_config_path)
    # predict = uv.predict()
    # print(predict)
    
    #unlearn
    # unlearn_config_path = './output/config/dataToPython_unlearn.json'
    # uv = UnlearnableIMCFN(unlearn_config_path)
    # uv.unlearn()
    