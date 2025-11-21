"""
VGG16 - VGG16 Neural Network for Malware Classification

This module implements a customized VGG16 neural network for malware classification.
It uses transfer learning with a pre-trained VGG16 model and fine-tunes specific layers
for binary classification (malware vs benignware).

Author: cchunhuang
Date: 2024
"""

import os
import time
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from Logger import setup_logger

class VGG16:
    def __init__(self, batch_size=4, learning_rate=5e-6, degrees=None, width_shift=0.0, 
                 height_shift=0.0, scale=None, shear=None, fill=None, horizontal_flip=0):
        """
        Initialize the VGG16 model with specified hyperparameters and data augmentation settings.
        
        Args:
            batch_size: Number of samples per batch
            learning_rate: Learning rate for optimizer
            degrees: Range for random rotation (default: [0, 0])
            width_shift: Horizontal shift range (fraction of total width)
            height_shift: Vertical shift range (fraction of total height)
            scale: Scale range for random zoom (default: [1.0, 1.0])
            shear: Shear range for random shear transformation (default: [0, 0, 0, 0])
            fill: Pixel fill value for areas outside boundaries
            horizontal_flip: Probability of random horizontal flip
        """
        self.logger = setup_logger("VGG16")
        self.logger.info('Constructing VGG16 object')
        
        # Initialize mutable default arguments
        if degrees is None:
            degrees = [0, 0]
        if scale is None:
            scale = [1.0, 1.0]
        if shear is None:
            shear = [0, 0, 0, 0]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Device: %s', self.device)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize attributes that will be set later
        self.input_data = None
        self.train_loader = None
        self.valid_loader = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        tf_list = [
            transforms.Resize((224, 224)),
            transforms.RandomAffine(
                degrees=degrees,  # rotation range 
                translate=(width_shift, height_shift),  # width shift and height shift
                scale=scale,  # zoom range
                shear=shear,  # shear range
                fill=fill  # Fill mode
            ),
            transforms.RandomHorizontalFlip(horizontal_flip),  # Horizontal flip
            transforms.ToTensor()   # rescale to 0~1
        ]
        
        self.tf = transforms.Compose(tf_list)
        self.tf_image = transforms.Compose([transforms.ToPILImage()] + tf_list)
        
    def loadDataFromFolder(self, input_path: str):
        """
        Load data from a folder organized by class labels.
        
        Args:
            input_path: Path to folder containing subfolders for each class
        """
        self.logger.info('Loading data from %s', input_path)
        self.input_data = ImageFolder(input_path, transform=self.tf)
        
    def loadDataFromImageFolder(self, input_data: ImageFolder):
        """
        Load data from an ImageFolder object.
        
        Args:
            input_data: Pre-constructed ImageFolder dataset
        """
        self.logger.info('Loading data from ImageFolder')
        self.input_data = input_data
        
    def loadDataFromArray(self, input_data: np.ndarray, label: np.ndarray):
        """
        Load data from numpy arrays.
        
        Args:
            input_data: Array of image data (N x H x W x C)
            label: Array of labels (N,)
        """
        self.logger.info('Loading data from array')
        images = input_data.astype(np.uint8)
        images = [self.tf_image(image) for image in images]
        self.input_data = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(label.astype(np.int64), dtype=torch.int64))
        
    def splitTrainData(self, train_ratio=0.8):
        """
        Split data into training and validation sets.
        
        Args:
            train_ratio: Fraction of data to use for training (default: 0.8)
        """
        self.logger.info('Split train data')
        train_size = int(train_ratio * len(self.input_data))
        val_size = len(self.input_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(self.input_data, [train_size, val_size])
        
        # set data loader
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size,shuffle=True)
          
    def loadModel(self, pretrained: str=None):
        """
        Load VGG16 model with customized classifier layers.
        
        The model uses transfer learning with frozen feature extraction layers
        and trainable final convolutional and classifier layers.
        
        Args:
            pretrained: Path to pre-trained model checkpoint (optional)
        """
        self.logger.info('Loading VGG16 model')
        # set model
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # customize model classifier
        self.model.classifier[0] = torch.nn.Linear(in_features=25088, out_features=2048, bias=True)
        self.model.classifier[1] = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.model.classifier[3] = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
        del self.model.classifier[6]
        del self.model.classifier[5]
        del self.model.classifier[4]
        
        # frozen layers, set requires_grad to False, so that the parameters will not be updated during training
        for param in self.model.parameters():
            param.requires_grad_(False)

        # unfrozen layers, set requires_grad to True, so that the parameters will be updated during training
        self.model.features[24].requires_grad_(True)
        self.model.features[26].requires_grad_(True)
        self.model.features[28].requires_grad_(True)
        self.model.classifier[0].requires_grad_(True)
        self.model.classifier[1].requires_grad_(True)
        self.model.classifier[3].requires_grad_(True)
        
        # set optimizer and loss function
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # load pretrained model
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.criterion.load_state_dict(checkpoint['loss'])
            self.model.eval()
        # summary(self.model, input_size=(self.batch_size, 3, 224, 224))
        
    def trainModel(self, epochs, log_path: str=None, model_path: str=None, print_information: str=""):
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of training epochs
            log_path: Path to save training logs (optional)
            model_path: Path to save trained model (optional)
            print_information: Additional information to log
            
        Returns:
            Dictionary containing training metrics including accuracy, precision,
            recall, F1-score, and training/validation history
        """
        self.logger.info('Training model')
        self.logger.info(print_information)
        
        if log_path is not None:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(print_information + '\n\n'
                        'Hyperparameters: ' + '\n' + 
                        'batch_size: ' + str(self.batch_size) + '\n' +
                        'learning_rate: ' + str(self.learning_rate) + '\n' +
                        'epochs: ' + str(epochs) + '\n' + '\n'
                        )
        
        self.model = self.model.to(self.device)
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        train_losses, valid_losses = [], []
        train_acc, valid_acc = [], []
        
        for epoch_num in range(epochs):
            start_time = time.time()
            iteration, iteration2 = 0, 0 # iteration count, used to calculate the average loss
            correct_train, total_train = 0, 0
            correct_valid, total_valid = 0, 0
            train_loss, valid_loss = 0.0, 0.0

            self.model.train() # set the model to training mode
            self.logger.info('epoch: %d / %d', epoch_num + 1, epochs)  
            
            # ---------------------------
            # Training Stage
            # ---------------------------
            for x, label in tqdm(self.train_loader, ncols=50) :
                x, label = x.to(self.device), label.to(self.device)
                self.optimizer.zero_grad() # clear the gradients of all optimized variables
                train_output = self.model(x) # forward pass: compute predicted outputs by passing inputs to the model
                train_loss_c = self.criterion(train_output, label) # calculate the batch loss
                train_loss_c.backward() # backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step() # perform a single optimization step (parameter update)
                
                # calculate training accuracy (correct_train / total_train)
                _, predicted = torch.max(train_output.data, 1) # get the predicted class
                total_train += label.size(0) # update the total count
                correct_train += (predicted == label).sum() # update the correct count
                train_loss += train_loss_c.item() # update the loss
                iteration += 1 # update the iteration count
                
                # possitive: malware, negative: benignware(label = 0)
                if epoch_num == epochs - 1:
                    for i in range(len(label)):
                        if label[i] == predicted[i]:
                            if label[i] == 1:
                                TP += 1
                            else:
                                TN += 1
                        elif label[i] == 0:
                            FN += 1
                        else:
                            FP += 1
                        
            self.logger.info('Training acc: %.3f | loss: %.3f', correct_train / total_train, train_loss / iteration)
            
            # if output_folder is not None:
            #     output_path = os.path.join(output_folder, 'epoch' + str(epoch + 1) + '.pth')
            #     self.saveModel(output_path)
            
            # --------------------------
            # Validation Stage
            # --------------------------
            self.model.eval() # set the model to evaluation mode
            for x, label in tqdm(self.valid_loader, ncols=50) :
                with torch.no_grad(): # turn off gradients for evaluation
                    x, label = x.to(self.device), label.to(self.device)
                    valid_output = self.model(x) # forward pass: compute predicted outputs by passing inputs to the model
                    valid_loss_c = self.criterion(valid_output, label) # calculate the batch loss
                    
                    # calculate validation accuracy (correct_valid / total_valid)
                    _, predicted = torch.max(valid_output.data, 1)
                    total_valid += label.size(0)
                    correct_valid += (predicted == label).sum()
                    valid_loss += valid_loss_c.item()
                    iteration2 += 1
                    
                    if epoch_num == epochs - 1:
                        for i in range(len(label)):
                            # possitive: benignware(label = 0), negative: malware
                            if label[i] == predicted[i]:
                                if label[i] == 0:
                                    TP += 1
                                else:
                                    TN += 1
                            elif predicted[i] == 0:
                                FP += 1
                            else:
                                FN += 1
            
            self.logger.info('Validation acc: %.3f | loss: %.3f', correct_valid / total_valid, valid_loss / iteration2)
                                            
            train_acc.append((correct_train / total_train).cpu().tolist()) # training accuracy
            valid_acc.append((correct_valid / total_valid).cpu().tolist())    # validation accuracy
            train_losses.append((train_loss / iteration))                    # train loss 
            valid_losses.append((valid_loss / iteration2))    # validate loss

            end_time = time.time()
            self.logger.info('Cost %.3f(secs)', end_time - start_time)
            
            if log_path is not None:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write('epoch: ' + str(epoch_num + 1) + ' / ' + str(epochs) + '\n')
                    f.write('Training acc: %.3f | loss: %.3f' % (correct_train / total_train, train_loss / iteration) + '\n')
                    f.write('Validation acc: %.3f | loss: %.3f' % (correct_valid / total_valid, valid_loss / iteration2) + '\n')
                    f.write('Cost %.3f(secs)' % (end_time - start_time) + '\n' + '\n')
        
        if model_path is not None:
            self.saveModel(model_path)
            
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "train_acc": train_acc, "valid_acc": valid_acc, "train_loss": train_losses, "valid_loss": valid_losses, "model_path": model_path}
    
    def saveModel(self, checkpoint_path: str):
        """
        Save the model checkpoint.
        
        Args:
            checkpoint_path: Path to save the model checkpoint
        """
        self.logger.info('Saving model to %s', checkpoint_path)
        torch.save({
                # 'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion.state_dict()
                }, checkpoint_path)
        
    def predict(self, input_data: np.ndarray):
        """
        Predict labels for input data.
        
        Args:
            input_data: Array of images to classify
            
        Returns:
            List of predicted class indices
        """
        self.logger.info('Predicting')
        
        self.model = self.model.to(self.device)
        self.model.eval() # set the model to evaluation mode
        
        result = []
        for data in input_data:
            data_tensor = self.tf_image(data.astype(np.uint8)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                result += [self.model(data_tensor).argmax().item()]
            
        return result
    
if __name__ == '__main__':
    input_folder = "./output/vectorize/detection"
    output_folder = "./DataAugmentation/"
    os.makedirs(output_folder, exist_ok=True)
    
    epoch = 50
    vgg16 = VGG16(4, 5e-6)
    
    logging_config = "./src/config/logging_config.json"
    vgg16.logger = setup_logger("VGG16", logging_config)
    
    vgg16.loadDataFromFolder(input_folder)
    vgg16.splitTrainData()
    vgg16.loadModel()
    result1 = vgg16.trainModel(epoch, os.path.join(output_folder, 'log.txt'), os.path.join(output_folder, 'model1.pth'), 'Normal Training')
    
    print('train_acc:', result1["train_acc"])
    print('valid_acc:', result1["valid_acc"])
    print('train_losses:', result1["train_loss"])
    print('valid_losses:', result1["valid_loss"])