
import numpy as np
from gpiozero import Button
from picamera import PiCamera
import matplotlib.pyplot as mp
import matplotlib.image as mi


# loading function from my other program!!
from mnistconverter import convert
from neuralnet import neuralNetwork


# ---------------------------- NUMBER FUNCTION ---------------------------------
# ----------------------------------------------------------------------------

def takephoto():
    camera.capture('/home/pi/Pictures/photos1/testimage.jpg')
    mp.imshow(mi.imread('/home/pi/Pictures/photos1/testimage.jpg'))
    mp.show()
    print('photo taken')
    print(predictImage())

def predictImage():
    f = "/home/pi/Pictures/photos1/testimage.jpg"
    image = convert(f,1, True)
    res, ret = ANN.run(image.flatten())
    return(res.argmax())

# ----------------------------- LOAD BASICS  ---------------------------------
# ----------------------------------------------------------------------------

ANN = neuralNetwork(noInputs = 784,noOutputs = 10, hiddenSize=200, learningRatex=0.1)
# set inputs to be 784 (image pixel size), outputs to be 10 (digits 0-9), size of hidden layer, and learning rate
ANN.loadWeights('weights.npz')

camera = PiCamera()
button = Button(4)

# ----------------------------- LOOP -------------------------------------------
# ------------------------------------------------------------------------------

while(True):
    button.when_pressed = takephoto