# Imports
import numpy as np
from PIL import Image


class RotateImage(object):
    '''
        Rotates the image about the centre of the image.
    '''

    def __init__(self, degrees):
        '''
            Arguments:
            degrees: rotation degree.
        '''

        # Write your code here
        self.degrees = degrees

    def __call__(self, sample):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here
        if not isinstance(sample, np.ndarray):
            img_new = sample.rotate(self.degrees, Image.NEAREST, expand=1)
            return img_new
        else:
            image = Image.fromarray(sample)
            img_new = image.rotate(self.degrees, Image.NEAREST, expand=1)
            return np.asarray(img_new)