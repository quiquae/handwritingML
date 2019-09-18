# neural network program!!
# USES 'MNIST DATASET'- dataset of handwritten digits downloaded from mnist database

import matplotlib.pyplot as plt 
import numpy as np
import random
import math

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
    

# ----------------------------------------------------------------------------
# -------------------------------- LOAD DATA ---------------------------------
# ----------------------------------------------------------------------------
print("Loading data...")
# datapath where my files for the testing/training data are stored in my computer
data_path = "C:\\Users\\creag\\Downloads\\"
# test image I took from my phone of my own handwriting!! for later
image = mnistconvert("C:\\Users\\creag\\OneDrive\\compsci-proj\\actual_image.jpeg",4,True)

# load test & train data from csv files
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",")
fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])
print("Done loading data.\n\n")

#print(train_imgs.shape)
#print(train_labels.shape)

lr = np.arange(10)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

#print(train_labels_one_hot[0:5])

# ----------------------------------------------------------------------------
# ------------------------------- RUN NETWORK --------------------------------
# ----------------------------------------------------------------------------
	
## FORMAT:: comes as comma seperated text file
#w/ first number being number it's supposed to be, then 784 pixel values 0-255
# so uploaded both test/train files into arrays of 785, first being number and rest being image
print("Training Network...")

ANN = neuralNetwork(noInputs = 784,noOutputs = 10, hiddenSize=200, learningRatex=0.1)
# set inputs to be 784 (image pixel size), outputs to be 10 (digits 0-9), size of hidden layer, and learning rate
ANN.initWeights() # initialize weights

# mnist - 28x28 image, pixels 0-255

## as we load a line (1 training image) we train the network on it
# for line in open(data_path + "mnist_train.csv"):
    # data_array = np.fromstring(line, sep=",")
    # converted = convertForm(data_array, ANN.numOutputs)
    # ANN.train(converted[0],converted[1])
	
for i in range(len(train_imgs)):
	ANN.train(train_imgs[i], train_labels_one_hot[i])

print("Training complete.\n")
      
## we load the test data as an array of arrays then check so we get an average  
#test_data_images = np.empty([10000,784])
#test_data_scores = np.empty([10000,10])
#row = 0

#load line as converted data into test array
# for line in open(data_path + "mnist_test.csv"):
    # test = convertForm(np.fromstring(line, sep=","), ANN.numOutputs)
    # #test_data_images.append(test[0])
    # #test_data_scores.append(test[1])
    # test_data_img = test[0]
    # test_data_label = test[1]
    # res = ANN.run(test_data_img)
    #print(res)
    #row += 1

# print("ACCURACY: "+str(ANN.check(test_data)))

print("Running Network...")

print("EXAMPLES::\n")
print("Running on training data (high accuracy - predicted labels should all match actual targets)")
for i in range(5):
	train_img = train_imgs[i]
	train_label = train_labels[i]
	res, ret = ANN.run(train_img)
	#print(res)
	print("Prediction = ", res.argmax())
	print("Target = ", train_label)

print("Running on test data")
correct  = 0
for i in range(len(test_labels)):
    test_img = test_imgs[i]
    test_label = test_labels[i]
    res, ret = ANN.run(test_img)
    if(i<5):
        print("Prediction = ", res.argmax()," ; Target = ", test_label)
    if(res.argmax()==test_label):
        correct+=1
        
print("Accuracy: ",str(100*correct/len(test_labels))," %")    

print("Visualize Test predictions: ")
#Plots predicitons/actual with green/red to show as correct or incorrect
nplots = 90 # plots a few (90) examples of our predictions/dataset
fig = plt.figure(figsize = (30,30))
fig.tight_layout()
n = random.randint(0,len(test_labels)-nplots) # random starting number so images plotted are relatively random
for j in range(n, nplots+n):
    plt.subplot(10,9,(j-n+1)) # divides window into smaller plots
    plt.imshow(np.reshape(test_imgs[j],(28,28)), cmap='binary') # show images!
    prediction  = ANN.run(test_imgs[j])[0].argmax() # see what our neural net predicts
    if(prediction==test_labels[j]): # if correct, hooray.
        x = plt.imread("C:\\Users\\creag\\OneDrive\\compsci-proj\\green_background2.png") # green = win
        plt.title(str(int(test_labels[j])))
    else:
        x = plt.imread("C:\\Users\\creag\\OneDrive\\compsci-proj\\red_background2.png") # red X = lose
        plt.title(str(prediction)+" X, "+str(int(test_labels[j]))+" ✓")
    plt.imshow(x)
    plt.axis('off')
    plt.subplots_adjust(hspace = 2)
plt.show() # show plot


print("Done running.")
