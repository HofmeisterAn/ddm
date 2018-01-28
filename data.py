#!/usr/bin/env pytho

'''This class represents a default data input modle.'''

class Data:

    def __init__(self, filePath):
        self.set_file_path(filePath)

    def get_file_path(self):
        return self.filePath

    def set_file_path(self, filePath):
        self.filePath = filePath

    def to_array(self):
        raise NotImplementedError("Please Implement this method.")

    def get_data_x(self):
        raise NotImplementedError("Please Implement this method.")

    def get_data_y(self):
        raise NotImplementedError("Please Implement this method.")
