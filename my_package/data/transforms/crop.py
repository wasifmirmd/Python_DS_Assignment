# Imports
from PIL import Image
import numpy as np


class CropImage(object):
    '''
        Performs either random cropping or center cropping.
    '''

    def __init__(self, shape, crop_type='center'):
        '''
            Arguments:
            shape: output shape of the crop (h, w)
            crop_type: center crop or random crop. Default: center
        '''

        # Write your code here
        self.shape = shape
        self.crop_type = crop_type
        if crop_type == 'center':
            self.is_center = True
        else:
            self.is_center = False

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here
        if isinstance(image, np.ndarray):
            img_copy = image.copy()
        else:
            img_copy = np.asarray(image)  # converting image to numpy array
        img_copy2 = img_copy.copy()
        new_height, new_width = self.shape
        if self.is_center:
            original_height = img_copy2.shape[0]
            original_width = img_copy2.shape[1]
            left = (original_width - new_width) / 2
            right = (original_width + new_width) / 2
            top = (original_height - new_height) / 2
            bottom = (original_height + new_height) / 2
            left, top = round(max(0, left)), round(max(0, top))
            right, bottom = round(min(original_width - 0, right)), round(min(original_height - 0, bottom))
            image_form = Image.fromarray(img_copy2)
            crop_image = image_form.crop(left, top, right, bottom)
            if isinstance(image, np.ndarray):
                return np.asarray(crop_image)
            else:
                return crop_image
        else:
            original_height = img_copy2.shape[0]
            original_width = img_copy2.shape[1]
            for i in range(original_height):
                for j in range(original_width):
                    if i <= new_height and j <= new_width:
                        continue
                    else:
                        img_copy2[i][j][0] = 255
                        img_copy2[i][j][1] = 255
                        img_copy2[i][j][2] = 255
            if isinstance(image, np.ndarray):
                return img_copy2
            else:
                return Image.fromarray(img_copy2)