# Imports
import numpy as np
from PIL import Image, ImageFilter


class BlurImage(object):
    '''
        Applies Gaussian Blur on the image.
    '''

    def __init__(self, radius):
        '''
            Arguments:
            radius (int): radius to blur
        '''

        # Write your code here
        self.radius = radius

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL Image)

            Returns:
            image (numpy array or PIL Image)
        '''

        # Write your code here
        if isinstance(image, np.ndarray):
            original_img = Image.fromarray(image)
            blur_intensity = self.radius
            blur_img = original_img.filter(ImageFilter.GaussianBlur(blur_intensity))
            np_array = np.asarray(blur_img)
            return np_array
        else:
            blur_intensity = self.radius
            blur_img = image.filter(ImageFilter.GaussianBlur(blur_intensity))
            return blur_img