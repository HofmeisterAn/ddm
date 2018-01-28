#!/usr/bin/env pytho

'''This class represents a default data input modle for images.'''

import os
import numpy

import matplotlib.pyplot

class Plot:

    def __init__(self):
        self.dir = 'result'
        self.cvs_loss = 'cnn-model-loss.csv'
        self.cvs_loss_validation = 'cnn-model-loss-validation.csv'

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_loss_report_path(self):
        return os.path.join(self.dir, self.cvs_loss)

    def get_loss_validation_report_path(self):
        return os.path.join(self.dir, self.cvs_loss_validation)

    def save(self, filePath, data):
        numpy.savetxt(filePath, data)

    def save_loss(self, data):
        self.save(self.get_loss_report_path(), data)

    def save_loss_validation(self, data):
        self.save(self.get_loss_validation_report_path(), data)

    def plot_loss(self):
        loss = numpy.loadtxt(self.get_loss_report_path())
        validation_loss = numpy.loadtxt(self.get_loss_validation_report_path())

        matplotlib.pyplot.plot(loss, linewidth=3, label='train')
        matplotlib.pyplot.plot(validation_loss, linewidth=3, label='valid')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.legend()
        matplotlib.pyplot.xlabel('epoch')
        matplotlib.pyplot.ylabel('loss')
        matplotlib.pyplot.yscale('log')
        matplotlib.pyplot.show()
