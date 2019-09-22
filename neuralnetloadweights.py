import matplotlib.pyplot as plt 
import numpy as np
import random
import math
import json

# loading function from my other program!!
from mnistconverter2 import mnistconvert

# ----------------------------------------------------------------------------
# ----------------------------- CLASSES & FUNCTIONS --------------------------
# ----------------------------------------------------------------------------

# each node sums all inputs , runs through function, + returns
# how to change the weights + biases as part of training
# passing image in, getting output
# check function
# initialize network (self)

## data comes in dataset as arrays of 785 length, need to be seperated into 784 pixel image array
    ## and targetOutput number
def convertForm(dataArray,numOutputs):
    targetOutput = int(dataArray[0])
    # take a number i.e. 7 which is target output, turn to [0,0,0,...1,0,0,0] for output nodes, where the 7th node is the only one which should be 1 (certain)
    targetOutputArray = np.zeros(numOutputs)
    targetOutputArray[targetOutput] = 1
    image = dataArray[1:]
    return image, targetOutputArray

# sigmoid function, which makes sure sum result x is between 0 and 1 / nonlinear idk? maths!!
def sigmoid(x):
    #return(1/(1+pow(np.e,-x)))
    return 1 / (1 + np.e ** -x)

class neuralNetwork:
    ## accepts parameters in constructor
    def __init__(self,noInputs,noOutputs,hiddenSize,learningRatex):
        self.numInputs = noInputs # number of inputs(image size)
        self.numOutputs = noOutputs # number of outputs (0-10 as digits)
        self.numHidden = hiddenSize # number of nodes in hidden layer
        self.learningRate = learningRatex # learning rate
        
    ## randomly initialize weights
    def initWeights(self):
        # randomly initializing weights, but still need mathematical upper/lower bound for randomness
        upper_bound = 1/(math.sqrt(self.numInputs))
        lower_bound = -upper_bound
        # reason to add +1 to inputs/hidden is because of bias, added to each input node in input to hidden, and each hidden node in hidden to output
        self.inputToHiddenWeights = np.random.normal(0,1,((self.numHidden,self.numInputs+1))) ## 2D array of the weights for input passing to hidden layer
        # weights for every node in array of this size randomly according to normal distribution with center 0 and spread 1 
        self.hiddenToOutputWeights = np.random.normal(0,1,((self.numOutputs,self.numHidden+1))) ## 2D array of the weights for hidden passing to output layerinput
                                                      
    ## train against imageInput by adjusting nodes so it reaches target
    def train(self, imageInput, targetOutput):
        targetOutput = np.array(targetOutput, ndmin=2).T
        # run the image input to know what the actual output the network predicts is
        actualOutput, hiddenOutput = self.run(imageInput)
        imageInput = np.concatenate((imageInput, [1]))
        imageInput = np.array(imageInput, ndmin=2).T
        # find error for each node by difference between target and actual output
        outputErrors = targetOutput - actualOutput
        # backpropogation: take an error, multiply by weight to find how important the weight is in contributing to error to attribute error, propogate back to attribute weights
        temp = outputErrors * actualOutput * (1.0 - actualOutput)
        #hiddenOutput = self.runHidden(imageInput)
        #temp = np.reshape(temp, (1,len(temp)))
		
        #temp = self.learningRate * np.dot(np.reshape(hiddenOutput.T, (len(hiddenOutput.T),1)),temp)
        temp = self.learningRate * np.dot(temp, hiddenOutput.T)
        #self.hiddenToOutputWeights += np.reshape(temp,(len(temp[0]),len(temp))) # add the error changes to our hidden weights
        self.hiddenToOutputWeights += temp # add the error changes to our hidden weights
        
        hiddenErrors = np.dot(self.hiddenToOutputWeights.T, outputErrors)
        temp = hiddenErrors * hiddenOutput * (1.0 - hiddenOutput)
        inputErrors = self.learningRate * np.dot(temp, imageInput.T)[:-1,:]
        self.inputToHiddenWeights += inputErrors # add the error changes to our input weights 
        

    ## runs input through network and returns output / accept input and pass through network to produce output                                              
    def run(self, imageInput):
        imageInput = np.concatenate((imageInput, [1]))
        imageInput = np.array(imageInput, ndmin=2).T
        #imageInput = np.append(imageInput,1) # add 1 at the end for bias node, so that it can multiply by its own weight and sum at the end
        # output from image input to hidden nodes is image input multiplied by weights then summed (dot product)
        hiddenOutput = np.dot(self.inputToHiddenWeights,imageInput)
        # sigmoid so that hidden output is between 0 and 1 (makes it nonlinear)
        hiddenOutput = sigmoid(hiddenOutput)
        hiddenOutput = np.concatenate((hiddenOutput, [[1]])) # add 1 at the end for bias node, so that it can multiply by its own weight and sum at the end
        # output from hidden input to actual output is hidden input multiplied by weights then summed (dot product)
        actualOutput = np.dot(self.hiddenToOutputWeights,hiddenOutput)
        # sigmoid so that actual output is nonlinear)
        actualOutput = sigmoid(actualOutput)
        return actualOutput, hiddenOutput
                                          
    ## runs input through network to return hidden layer output
    def runHidden(self, imageInput):
        #imageInput = np.append(imageInput,1) # add 1 at the end for bias node, so that it can multiply by its own weight and sum at the end
        imageInput = np.concatenate((imageInput, [1]))
		# output from image input to hidden nodes is image input multiplied by weights then summed (dot product)
        hiddenOutput = np.dot(self.inputToHiddenWeights,imageInput)
        # sigmoid so that hidden output is between 0 and 1 (makes it nonlinear)
        hiddenOutput = sigmoid(hiddenOutput)
        return(hiddenOutput)

    ## checks/ returns error rate
    def check(self,imageInput,targetOutput):
        # find error for each node by difference between target and actual output
        outputErrors = targetOutput - actualOutput
        return(outputErrors/len(targetOutput))

    # get weights - useful for storing neural network so no training needed
    def getWeights(self):
        return(self.inputToHiddenWeights, self.hiddenToOutputWeights)
    
    def setWeights(self,iH, hO):
        self.inputToHiddenWeights = iH
        self.hiddenToOutputWeights = hO

# ----------------------------------------------------------------------------
# ----------------------------- LOAD WEIGHTS ---------------------------------
# ----------------------------------------------------------------------------
print("Training Network...")

ANN = neuralNetwork(noInputs = 784,noOutputs = 10, hiddenSize=200, learningRatex=0.1)
# set inputs to be 784 (image pixel size), outputs to be 10 (digits 0-9), size of hidden layer, and learning rate
ANN.initWeights() # initialize weights
file = open("neuralnetweights.json", "r")
x,y = json.load(file)
ANN.setWeights(x,y)
print("Check image")
# test image I took from my phone of my own handwriting!! for later
image = mnistconvert("C:\\Users\\creag\\OneDrive\\compsci-proj\\actual_image.jpeg",4,True)
print(ANN.run(image)[0].argmax())
print("Done running.")
