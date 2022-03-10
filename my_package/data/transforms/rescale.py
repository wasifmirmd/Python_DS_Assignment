# Imports
from PIL import Image
import numpy as np


class RescaleImage(object):
    '''
        Rescales the image to a given size.
    '''

    def __init__(self, output_size):
        '''
            Arguments:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        '''
        self.is_tuple = isinstance(output_size, tuple)
        self.output_size = output_size
        # Write your code here

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)

            Note: You do not need to resize the bounding boxes. ONLY RESIZE THE IMAGE.
        '''

        # Write your code here
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            if not self.is_tuple:
                height = image.shape[0]
                width = image.shape[1]
                aspect_ratio = width / height
                if height < width:
                    img_resize = img.resize((int(self.output_size * aspect_ratio), self.output_size))
                else:
                    img_resize = img.resize((self.output_size, int(self.output_size / aspect_ratio)))
                return np.asarray(img_resize)
            else:
                img_resize = img.resize(self.output_size)
                return np.asarray(img_resize)
        else:
            if not self.is_tuple:
                img_np = np.asarray(image)
                height = img_np.shape[0]
                width = img_np.shape[1]
                aspect_ratio = width / height
                if height < width:
                    img_resize = image.resize((self.output_size, int(self.output_size * aspect_ratio)))
                else:
                    img_resize = image.resize((int(self.output_size / aspect_ratio), self.output_size))
                return img_resize
            else:
                image_resize = image.resize(self.output_size)
                return image_resize