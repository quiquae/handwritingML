import matplotlib.pyplot as plt 
import numpy as np
import random
import math
import json,codecs


# loading function from my other program!!
from mnistconverter import convert
from neuralnet import neuralNetwork

# ----------------------------- LOAD WEIGHTS ---------------------------------
# ----------------------------------------------------------------------------

ANN = neuralNetwork(noInputs = 784,noOutputs = 10, hiddenSize=200, learningRatex=0.1)
# set inputs to be 784 (image pixel size), outputs to be 10 (digits 0-9), size of hidden layer, and learning rate
ANN.loadWeights('weights.npz')

images = ["actual_image.jpeg","actual_image2.jpeg","actual_image3.jpg"]
for i in range(0,3):
    f = "data/"+images[i]
    image = convert(f,4,True)
    res, ret = ANN.run(image.flatten())
    print("Prediction = ", res.argmax())

