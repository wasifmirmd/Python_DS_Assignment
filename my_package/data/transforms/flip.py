# Imports
import numpy as np
from PIL import Image

class FlipImage(object):
    '''
        Flips the image.
    '''

    def __init__(self, flip_type='horizontal'):
        '''
            Arguments:
            flip_type: 'horizontal' or 'vertical' Default: 'horizontal'
        '''

        # Write your code here
        self.flipHorizontal = (flip_type == 'horizontal')

    def __call__(self, image):
        '''
            Arguments:
            image (numpy array or PIL image)

            Returns:
            image (numpy array or PIL image)
        '''

        # Write your code here
        if not isinstance(image, np.ndarray):
            imgMed = np.asarray(image)
        else:
            imgMed = image.copy()
        img = imgMed.copy()
        height = img.shape[0]
        width = img.shape[1]
        if self.flipHorizontal:
            for i in range(int(width / 2)):
                for j in range(height):
                    img[j][i][0], img[j][width - 1 - i][0] = img[j][width - 1 - i][0], img[j][i][0]
                    img[j][i][1], img[j][width - 1 - i][1] = img[j][width - 1 - i][1], img[j][i][1]
                    img[j][i][2], img[j][width - 1 - i][2] = img[j][width - 1 - i][2], img[j][i][2]

        else:
            for j in range(int(height / 2)):
                for i in range(width):
                    img[j][i][0], img[height - 1 - j][i][0] = img[height - 1 - j][i][0], img[j][i][0]
                    img[j][i][1], img[height - 1 - j][i][1] = img[height - 1 - j][i][1], img[j][i][1]
                    img[j][i][2], img[height - 1 - j][i][2] = img[height - 1 - j][i][2], img[j][i][2]

        if isinstance(image, np.ndarray):
            return img
        else:
            return Image.fromarray(img)