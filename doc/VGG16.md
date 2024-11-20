# VGG16.py

## Overview

This script implements a customized VGG16 model for image classification tasks. It provides functionality for data loading, model training, evaluation, and prediction using PyTorch.

## Dependencies

- os
- time
- numpy
- tqdm
- torch
- torchvision
- Logger (custom module)

## Class: VGG16

### Constructor

```python
VGG16(batch_size=4, learning_rate=5e-6, degrees=[0, 0], width_shift=0.0, 
      height_shift=0.0, scale=[1.0, 1.0], shear=[0, 0, 0, 0], fill=None, horizontal_flip=0)
```

Initializes the VGG16 object with specified parameters for data augmentation and training.

#### Parameters:

- `batch_size` (int, default=4): The number of samples per batch to load.
- `learning_rate` (float, default=5e-6): The learning rate for the optimizer.
- `degrees` (list of two ints, default=[0, 0]): Range of degrees to select for random rotation. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
- `width_shift` (float, default=0.0): Fraction of total width to shift the image horizontally.
- `height_shift` (float, default=0.0): Fraction of total height to shift the image vertically.
- `scale` (list of two floats, default=[1.0, 1.0]): Scaling factor range for image size scaling.
- `shear` (list of four floats, default=[0, 0, 0, 0]): Shear angle in degrees in counter-clockwise direction for each side of the image.
- `fill` (int or tuple, default=None): Pixel fill value for the area outside the transformed image. If given a number, it is used for all bands respectively.
- `horizontal_flip` (float, default=0): Probability of flipping the image horizontally.

### Methods

#### `loadDataFromFolder(input_path: str)`

Loads data from the specified input path using torchvision's ImageFolder.

#### `loadDataFromImageFolder(input_data: ImageFolder)`

Loads data from a pre-existing ImageFolder object.

#### `loadDataFromArray(input_data: np.ndarray, label: np.ndarray)`

Loads data from numpy arrays of images and labels.

#### `splitTrainData(train_ratio=0.8)`

Splits the loaded data into training and validation sets.

#### `loadModel(pretrained: str=None)`

Loads and customizes the VGG16 model.

- If `pretrained` is None, it loads the default VGG16 model and customizes it.
- If `pretrained` is a string, it loads a previously saved model from the specified path.

In both cases, the method sets up the optimizer and loss function.

#### `trainModel(epochs, log_path: str=None, model_path: str=None, print_information: str="")`

Trains the model for the specified number of epochs.

- `epochs`: Number of training epochs.
- `log_path`: If provided, training logs will be written to this file. If None, no log file will be created.
- `model_path`: If provided, the trained model will be saved to this path. If None, the model won't be saved.
- `print_information`: Additional information to be printed/logged at the start of training.

This method performs both training and validation for each epoch, and returns various metrics including accuracy, precision, recall, and F1 score.

#### `saveModel(checkpoint_path: str)`

Saves the current state of the model to the specified path.

#### `predict(input_data: np.ndarray)`

Makes predictions on input data.

- `input_data`: A numpy array of images to be predicted.
- Returns a list of predicted class indices for each input image.


## Key Features

1. **Data Augmentation**: Supports various data augmentation techniques including rotation, shifting, scaling, shearing, and horizontal flipping.

2. **Flexible Data Loading**: Supports loading data from folders, ImageFolder objects, or numpy arrays.

3. **Model Customization**: The VGG16 model's classifier is customized for the specific task.

4. **Transfer Learning**: Most layers are frozen, with only the last few layers and the custom classifier being trainable.

5. **Training Metrics**: Calculates and returns various metrics including accuracy, precision, recall, and F1 score.

6. **Logging**: Supports optional logging of training progress to a file.

7. **Model Saving and Loading**: Allows optional saving of trained models and loading of pre-trained models.

8. **GPU Support**: Automatically uses CUDA if available.

## Usage Example

```python
input_folder = "./input_folder/"
output_folder = "./output_folder/"
os.makedirs(output_folder, exist_ok=True)

epoch = 50
vgg16 = VGG16(4, 5e-6)

vgg16.loadDataFromFolder(input_folder)
vgg16.splitTrainData()
vgg16.loadModel()  # Load default model
# vgg16.loadModel("path/to/pretrained/model.pth")  # Load pre-trained model

# Train with logging and model saving
result = vgg16.trainModel(epoch, log_path=os.path.join(output_folder, 'log.txt'), 
                           model_path=os.path.join(output_folder, 'model1.pth'), 
                           print_information='Normal Training')

# Train without logging or model saving
# result = vgg16.trainModel(epoch)
```

## Notes

- The model is designed for binary classification (2 output classes).
- The script uses SGD optimizer and CrossEntropyLoss.
- Training progress is displayed using tqdm progress bars.
- The model automatically switches between training and evaluation modes during the training process.
- Precision, recall, and F1 score are calculated assuming positive class is labeled as 1 and negative class as 0.
- When loading a pre-trained model, make sure the saved model's architecture matches the current model's architecture.
- Logging and model saving during training are optional and controlled by the `log_path` and `model_path` parameters in the `trainModel` method.

