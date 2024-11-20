# ImageGenerator.py

## Overview

`ImageGenerator` is a Python class designed to generate visual representations of binary files as images. It converts binary data into color-mapped images, which can be useful for data visualization or analysis purposes.

## Dependencies

- os
- cv2
- numpy
- tqdm
- matplotlib
- Logger (custom module)

## Class: ImageGenerator

### Attributes

- `width`: A dictionary mapping file sizes (in KB) to image widths. Keys are file sizes in bytes, values are corresponding image widths.
- `cm`: A colormap object from matplotlib, set to 'jet'.
- `logger`: A logger object for the class, set up using a custom `Logger` module.

### Methods

#### `__init__(self) -> None`

Initializes the ImageGenerator object, setting up the colormap and logger.

#### `image_width(self, vec_len) -> int`

Determines the appropriate image width based on the input vector length (file size).

- Parameters:
  - `vec_len` (int): Length of the input vector (file size in bytes)
- Returns:
  - int: The appropriate width for the image

#### `generateImage(self, input_paths: list, output_folder: str = None) -> dict`

Generates images from binary files.

- Parameters:
  - `input_paths` (list): A list of paths to input binary files
  - `output_folder` (str, optional): Path to the folder where generated images will be saved. If set to None, images will not be saved to disk.
- Returns:
  - dict: A dictionary mapping input file paths to generated image arrays

### Process

1. Reads binary data from each input file
2. Converts binary data to numpy arrays
3. Determines appropriate image width based on file size
4. Reshapes the array into a 2D matrix
5. Applies color mapping to create a 3D image array
6. Resizes the image to 224x224 pixels
7. If `output_folder` is provided, saves the generated images to the specified folder
8. Returns a dictionary of generated image arrays

## Usage Example

```python
generator = ImageGenerator()
input_files = ['file1.bin', 'file2.bin']
output_folder = 'output_images'
image_dict = generator.generateImage(input_files, output_folder)

# To generate images without saving them to disk:
image_dict = generator.generateImage(input_files)
```

This will generate images from 'file1.bin' and 'file2.bin'. If `output_folder` is specified, it will save the images in the 'output_images' folder. In both cases, it returns a dictionary containing the image arrays.

## Notes

- The class uses a 'jet' colormap by default for visualization.
- Generated images are resized to 224x224 pixels, regardless of the original binary file size.
- The class uses tqdm for progress visualization during image generation and saving.
- Logging is implemented to track the image generation and saving processes.
- If `output_folder` is None, the images will be generated but not saved to disk.
