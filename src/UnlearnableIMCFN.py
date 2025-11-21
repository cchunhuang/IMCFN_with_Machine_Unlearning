import os
import csv
import json
import time
import copy
import random

from malwareDetector.config import read_config, write_config_to_file
from UnlearningConfig import read_unlearning_config
from IMCFN import IMCFN
from Logger import setup_logger

FILE_NAME = 0
LABEL = 1

class UnlearnableIMCFN:
    def __init__(self, config_path: str=None):
        if config_path is None:
            self.config = read_unlearning_config()
        else:
            self.config = read_unlearning_config(config_path)
        
        # -----set paths-----
        
        self.model_folder = os.path.join(self.config.folder.model, time.strftime("%Y%m%d_%H%M", time.localtime()))
        
        self.subdetector_shard_folder = [os.path.join(self.model_folder, f"shard{i}") for i in range(self.config.model.shard)]
        for shard_folder in self.subdetector_shard_folder:
            os.makedirs(shard_folder, exist_ok=True)
            
        self.subdetector_label_path = os.path.join(self.model_folder, "subdetector_label.csv")
        self.subdetector_config_path = os.path.join(self.model_folder, "subdetector_config.json")
        
        # -----load logger-----
        
        self.config.path.log = os.path.join(self.model_folder, 'loging.log')
        self.logger = setup_logger("UnlearnableIMCFN", self.config.path.logging_config, self.config.path.log)
        self.logger.info('Constructing UnlearnableIMCFN object')
        
        # -----spilt data into shards and slices-----
        
        if self.config.train:
            if self.config.model.overwrite is False:
                self.config.path.position = os.path.join(self.model_folder, 'position.csv')
                self.config.path.subdetector_name = os.path.join(self.model_folder, 'subdetector_name.csv')
            
            labels = self.getLabel()
                
            SEED_VALUE = 42
            random.seed(SEED_VALUE)
            random.shuffle(labels)
                
            self.input_data = [[[] for j in range(self.config.model.slice)] for i in range(self.config.model.shard)]
            for i in range(0, len(labels), self.config.model.batch_size):
                shard_idx = i // self.config.model.batch_size % self.config.model.shard
                slice_idx = i // self.config.model.batch_size // self.config.model.shard % self.config.model.slice
                for j in range(i, min(i + self.config.model.batch_size, len(labels))):
                    self.input_data[shard_idx][slice_idx].append(labels[j])
            
            self.savePosition()
        
        #-----initialize IMCFN-----
        
        subdetector_config = copy.deepcopy(self.config)
        subdetector_config.path.del_param('position')
        subdetector_config.path.del_param('subdetector_name')
        subdetector_config.path.del_param('unlearn')
        subdetector_config.model.del_param('shard')
        subdetector_config.model.del_param('slice')
        
        subdetector_config.path.log = self.config.path.log
        
        write_config_to_file(subdetector_config, self.subdetector_config_path)
        
        self.imcfn = IMCFN(self.subdetector_config_path)
        
    def trainModel(self, shard_list: list=None, start_slice: list=None):
        '''
        Train the model.
        '''
        self.logger.info('Training model')
        
        # -----Validation-----
        
        if shard_list is None:
            shard_list = [i for i in range(self.config.model.shard)]
        else:
            if not 1 <= len(shard_list) <= self.config.model.shard:
                raise ValueError(f"shard_list size must be between 1 and {self.config.model.shard}")
            for item in shard_list:
                if not isinstance(item, int) or not 0 <= item < self.config.model.shard:
                    raise ValueError(f"All items in shard_list must be integers between 0 and {self.config.model.shard - 1}")
            
        if start_slice is None:
            start_slice = [0 for i in range(self.config.model.shard)]
        else:
            if len(start_slice) != self.config.model.shard:
                raise ValueError(f"start_slice size must be exactly {self.config.model.shard}")
            for item in start_slice:
                if not isinstance(item, int) or not 0 <= item < self.config.model.slice:
                    raise ValueError(f"All items in start_slice must be integers between 0 and {self.config.model.slice - 1}")
                
        # -----vectorize-----        
        
        self.imcfn.vectorize()
                
        # -----initialize subdetector name-----
        
        if not os.path.exists(self.config.path.subdetector_name):
            subdetector_name = [["DEF"] + [None for j in range(self.config.model.slice)] for i in range(self.config.model.shard)]
        else:                
            subdetector_name = self.getSubdetectorName()   
            
        # -----process config-----
        
        subdetector_config = read_config(self.subdetector_config_path)
        subdetector_config.path.input_files = self.subdetector_label_path
        
        # -----train model-----
        
        score = []
        
        for shard_idx in shard_list:
            # load pretrained model
            subdetector_config.path.pretrained = None if subdetector_name[shard_idx][start_slice[shard_idx]] == '' else subdetector_name[shard_idx][start_slice[shard_idx]]
            write_config_to_file(subdetector_config, self.subdetector_config_path)
            
            self.imcfn.model(False)
            
            subdetector_config.path.pretrained = None
            
            # get data
            data = [_data for _slice_idx in range(start_slice[shard_idx]) for _data in self.input_data[shard_idx][_slice_idx]]
            
            for slice_idx in range(start_slice[shard_idx], self.config.model.slice):
                data.extend(self.input_data[shard_idx][slice_idx])
                self.saveSubdetectorLabel(data)
                
                subdetector_name[shard_idx][slice_idx + 1] = os.path.join(self.subdetector_shard_folder[shard_idx], f"slice{slice_idx}.pth")
                
                subdetector_config.path.log = os.path.join(self.subdetector_shard_folder[shard_idx], "log.txt")
                subdetector_config.path.model = subdetector_name[shard_idx][slice_idx + 1]
                subdetector_config.model.print_information = f"*****Shard: {shard_idx} | Slice: {slice_idx}*****"
                write_config_to_file(subdetector_config, self.subdetector_config_path)
                
                result = self.imcfn.model(True)
                
                train_acc = [{"epoch": i, "accuracy": result['train_acc'][i]} for i in range(len(result['train_acc']))]
                train_loss = [{"epoch": i, "loss": result['train_loss'][i]} for i in range(len(result['train_loss']))]
                val_acc = [{"epoch": i, "accuracy": result['valid_acc'][i]} for i in range(len(result['valid_acc']))]
                val_loss = [{"epoch": i, "loss": result['valid_loss'][i]} for i in range(len(result['valid_loss']))]
                test_acc = result['test_result']['accuracy']
                
                score.append({"model_name": subdetector_name[shard_idx][slice_idx + 1], "train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc})
        
        # -----save model-----
        
        self.saveSubdetectorName(subdetector_name)
        
        # -----test model-----
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        if os.path.exists(self.config.path.test_files):
            subdetector_config.path.input_files = self.config.path.input_files = self.config.path.test_files
            write_config_to_file(subdetector_config, self.subdetector_config_path)
            
            test_predict = self.predict()
            test_labels = self.getLabel(self.config.path.test_files)
            test_labels = {test_labels[i][FILE_NAME]: test_labels[i][LABEL] for i in range(len(test_labels))}
            
            for test_dict in test_predict:
                true_label = test_labels[test_dict['name']]
                prediction = test_dict['detection']
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
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        final_result = {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
        
        #-----save score-----
        
        if self.config.model.overwrite is False:
            self.config.path.score = os.path.join(self.model_folder, "score.json")
        
        with open(self.config.path.score, 'w') as f:
            json.dump([{"final_result": final_result}] + score, f, indent=4)
        
    def predict(self):
        '''
        Predict the label of the file.
        '''
        self.logger.info('Predicting')
        
        self.imcfn.vectorize()
        
        file_names = self.getFileName()
        subdetector_name = self.getSubdetectorName()
        subdetector_config = read_config(self.subdetector_config_path)
        
        vote = {file_name: {} for file_name in file_names}
        
        LATEST_MODEL = -1
        
        for shard_idx in range(self.config.model.shard):
            if subdetector_name[shard_idx][LATEST_MODEL] == None or subdetector_name[shard_idx][LATEST_MODEL] == '':
                raise ValueError(f"Subdetector shard{shard_idx} is not trained")
            
            subdetector_config.path.pretrained = subdetector_name[shard_idx][LATEST_MODEL]
            write_config_to_file(subdetector_config, self.subdetector_config_path)
            
            self.imcfn.model(False)
            predict_result = self.imcfn.predict()
            for file_name in predict_result.keys():
                if predict_result[file_name] not in vote[file_name].keys():
                    vote[file_name][predict_result[file_name]] = 1
                else:
                    vote[file_name][predict_result[file_name]] += 1
        
        result = [{"name": file_name, "detection": max(vote[file_name], key=vote[file_name].get)} for file_name in predict_result.keys()]
        
        if self.config.model.overwrite is False:
            self.config.path.result = os.path.join(self.model_folder, "predict_result.json")
        
        with open(self.config.path.result, 'w') as f:
            json.dump(result, f, indent=4)
            
        return result
    
    def unlearn(self):
        '''
        Unlearn the model.
        '''
        self.logger.info('Unlearning')
        
        self.getPosition()
        file_names = self.getFileName()
        
        shard_list = set()
        start_slice = [self.config.model.slice - 1 for i in range(self.config.model.shard)]
        
        for unlearn_data in file_names:
            flag = False
            for shard_idx in range(self.config.model.shard):
                for slice_idx in range(self.config.model.slice):
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
                self.logger.warning(f'The file was not learned: {unlearn_data}')
        
        if self.config.model.overwrite is False:
            self.config.path.position = os.path.join(self.model_folder, 'position.csv')
            self.config.path.subdetector_name = os.path.join(self.model_folder, 'subdetector_name.csv')
            
        self.savePosition()
        
        shard_list = list(shard_list)
        if len(shard_list) != 0:
            self.logger.info(f'Retrain shard: {shard_list}')
            self.logger.info(f'Start slice: {start_slice}')
            
            with open(self.config.path.input_files, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['file_name', 'label'])
                for shard_idx in shard_list:
                    for slice_idx in range(self.config.model.slice):
                        for data in self.input_data[shard_idx][slice_idx]:
                            writer.writerow(data)
                    
            self.trainModel(shard_list, start_slice)
                
    def savePosition(self):
        '''
        Save the position of the data.
        '''
        with open(self.config.path.position, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'label', 'shard', 'slice'])
            for shard_idx in range(len(self.input_data)):
                for slice_idx in range(len(self.input_data[shard_idx])):
                    for data in self.input_data[shard_idx][slice_idx]:
                        writer.writerow([data[FILE_NAME], data[LABEL], shard_idx, slice_idx])
        
    def getPosition(self):
        '''
        Get the position of the data.
        '''
        if not os.path.exists(self.config.path.position):
            raise FileNotFoundError(f"File not found: {self.config.path.position}")
        
        SHARD_IDX = 2
        SLICE_IDX = 3
        
        self.input_data = [[[] for j in range(self.config.model.slice)] for i in range(self.config.model.shard)]
        with open(self.config.path.position, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for rows in reader:
                self.input_data[int(rows[SHARD_IDX])][int(rows[SLICE_IDX])].append([rows[FILE_NAME], rows[LABEL]])
        
    def saveSubdetectorLabel(self, data: list):
        '''
        Save the label of the subdetector.
        '''
        with open(self.subdetector_label_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'label'])
            for i in range(len(data)):
                writer.writerow(data[i])
        
    def saveSubdetectorName(self, model_name: list):
        '''
        Save the name of the subdetector.
        '''
        with open(self.config.path.subdetector_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['default'] + [f'slice{i}' for i in range(self.config.model.slice)])
            for i in range(self.config.model.shard):
                writer.writerow(model_name[i])       
                
    def getSubdetectorName(self):
        '''
        Get the name of the subdetector.
        '''
        with open(self.config.path.subdetector_name, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            return [rows for rows in reader]  
        
    def getLabel(self, path: str=None):
        '''
        Get label from csv file.
        '''
        if path is None:
            path = self.config.path.input_files
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            labels = []
            for rows in reader:
                if not os.path.exists(os.path.join(self.config.folder.dataset, rows[FILE_NAME])):
                    self.logger.warning(f"File not found: {os.path.join(self.config.folder.dataset, rows[FILE_NAME])}")
                    continue
                labels.append([rows[FILE_NAME], rows[LABEL]])
        return labels
    
    def getFileName(self):
        '''
        Get file name from csv file.
        '''
        with open(self.config.path.input_files, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            file_name = [rows[FILE_NAME] for rows in reader]
        return file_name
        
if __name__ == '__main__':
    # train
    config_path = './output/config/dataToPython.json'
    uv = UnlearnableIMCFN(config_path)
    uv.trainModel()
    
    #predict
    # config_path = './output/config/dataToPython_predict.json'
    # uv = UnlearnableIMCFN(config_path)
    # predict = uv.predict()
    # print(predict)
    
    #unlearn
    # config_path = './output/config/dataToPython_unlearn.json'
    # uv = UnlearnableIMCFN(config_path)
    # uv.unlearn()
    