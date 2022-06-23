#!/usr/bin/env python
import numpy as np
from model import *
import time

def main():
    # instantiate class and start loop function
    """params for the model class"""

    #switch between train mode and prediction mode (True means train), setting this param to False starts an ros service
    train_mode = False

    #toggle show plot
    show_plot = True

    #the learning rate
    alpha = 0.001

    optimizer = "GradientDescent"

    loss_function = "CEL"

    store_weights = True

    #this is the model class which holds all necessary parts
    model = Model(train_mode, alpha, optimizer, loss_function)
    epochs = 10

    if train_mode:

        start = time.time()
        model.train(epochs, show_plot)
        ende = time.time()

        print('Total time: {:5.3f}min'.format(((ende-start)/60.0)))

        if store_weights:
            model.store_model()
    else:
        model.predict()

if __name__=='__main__':
    main()
