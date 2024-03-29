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


#-------------------- TESTING PHOTO EXAMPLES -------------------------------
#----------------------------------------------------------------------------

## for the actual_image.jpeg images (actual iPhone camera photos) list of the values they are supposed to be: 
actual_imgs=[6, 0, 6, 3, 1, 2, 4, 5, 3, 3, 2, 9, 2, 4,5,4,1,5,5,7,4]
## for the test_image images (binary/black&white) list of the values they are supposed to be: 2, 4, 5, 6, 3

#images = ["actual_image.jpeg","actual_image2.jpeg","actual_image3.jpg"]
print("Image testing!")
c = 0
x = 0
t = 0
for i in range(1,10):
    f = "data/actual_image"+str(i)+".jpg"
    image = convert(f,1, False)
    res, ret = ANN.run(image.flatten())
    print(" prediction = ", res.argmax(), " actual = ", actual_imgs[i-1])
    if(actual_imgs[i-1]==res.argmax()):
        c+=1
    else:
        x+=1
    t+=1
print("correct: ",c," incorrect: ",x," out of total: ",t," images")
p = 100*round(((c+0.00005)/t),4)
print("% correct: ",p)

