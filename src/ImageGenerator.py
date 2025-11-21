import os
import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from Logger import setup_logger

class ImageGenerator:
    width = {10 * 2**10: 32, 30 * 2**10: 64, 60 * 2**10: 128, 100 * 2**10: 256, 200 * 2**10: 384, 500 * 2**10: 512, 1000 * 2**10: 768}  # bigger: 1024
    
    def __init__(self) -> None:
        self.cm = plt.get_cmap('jet')
        self.logger = setup_logger("ImageGenerator")
    
    def image_width(self, vec_len):
        '''
        Return the width of image based on file size.
        '''
        for byte in self.width.keys(): 
            if vec_len <= byte:        # vec_len <= byte KB
                return self.width[byte]
        return 1024
    
    def generateImage(self, input_paths: list, output_folder: str=None):
        '''
        Generate image from binary file.
        '''
        self.logger.info("Generating images")
        image_array = {}
        for input_path in tqdm(input_paths, ncols=50):
            with open(input_path, 'rb') as f:
                binary_vector = f.read()
                vector_array = np.frombuffer(binary_vector, dtype=np.uint8)
                
            width = self.image_width(len(vector_array))
            length = len(vector_array)//width
            vector_array = vector_array[:width*length].reshape(length, width)

            image_array[input_path] = cv2.resize(self.cm(vector_array, bytes=True), (224, 224)) # convert 2D array to 3D array using colormap
            
        if output_folder is not None:
            self.logger.info("Saving images")
            os.makedirs(output_folder, exist_ok=True)
            for file_name in tqdm(image_array.keys(), ncols=50):
                plt.imsave(os.path.join(output_folder, os.path.basename(file_name).split('.')[0]  + '.png'), image_array[file_name])
        
        # image_array[input_paths] = image(224*224*4)
        return image_array