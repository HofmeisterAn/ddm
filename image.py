#!/usr/bin/env pytho

'''This class represents a default data input modle for images.'''

import numpy

from PIL import Image

from data import Data

class ImageData(Data):

    def __init__(self, filePath, width, heigth, x, y):
        self.width = width
        self.height = heigth
        self.set_points(x, y)
        self.set_file_path(filePath)
        self.set_image(filePath) # Todo: Use only one method instead

    def get_file_path(self):
        return super(ImageData, self).get_file_path()

    def set_file_path(self, filePath):
        super(ImageData, self).set_file_path(filePath)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_image(self):
        return self.image

    def set_image(self, filePath):
        image = Image.open(filePath).convert('L') # Todo: Add convert grayscale as parameter too, change numpy array lenght in reshape from 1 to 3
        image = image.resize((self.get_width(), self.get_height()), Image.NEAREST)
        self.image = image
        self.image.load()

    def get_points(self):
        return self.x, self.y

    def set_points(self, x, y):
        self.x = x / 3024 # Origin image widht and heigth, maybe store value in metadata.json
        self.y = y / 3024

    def to_array(self):
        return numpy.array(self.get_image(), dtype='uint8') / 255  # Scale pixel values to [0, 1]

    def get_data_x(self):
        return self.to_array().reshape(1, self.get_width(), self.get_height(), 1) # Reshape array to (1, 255, 255, 1 v 3)

    def get_data_y(self):
        x, y = self.get_points()
        return numpy.array([[x, y, 0]]) # Two target values, this can be extended to more values
