#FOR CV2:: pylint args!!
# go to settings (bottom left cog)
#find Python > Linting: Pylint Args (search pylint args)
#add:
# --generate-members
# --extension-pkg-whitelist=cv2

import numpy as np
import random
import matplotlib.pyplot as plt
from neuralnet import neuralNetwork
from mnistconverter import convert
import argparse #pip install argparse

## data comes in dataset as arrays of 785 length, need to be seperated into 784 pixel image array
    ## and targetOutput number
def convertForm(dataArray, numOutputs):
    targetOutput = int(dataArray[0])
    # take a number i.e. 7 which is target output, turn to [0,0,0,...1,0,0,0] for output nodes, where the 7th node is the only one which should be 1 (certain)
    targetOutputArray = np.zeros(numOutputs)
    targetOutputArray[targetOutput] = 1
    image = dataArray[1:]
    return image, targetOutputArray

# ----------------------------------------------------------------------------
# -------------------------------- LOAD DATA ---------------------------------
# ----------------------------------------------------------------------------
print("Loading data...")
# datapath where my files for the testing/training data are stored in my computer
data_path = ""

print("Loading images.")
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

lr = np.arange(10)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

# ----------------------------------------------------------------------------
# ------------------------------- RUN NETWORK --------------------------------
# ----------------------------------------------------------------------------

niterations = 1 # number of times network will be run- best accuracy percentage network will be saved as weights for use later
maxacc = 0 # maximum accuracy counter (starts at 0, replaced by accuracies if larger)
show = False ## whether to show lots of information about & visualise each iteration or not

## repeats for the number of iterations
for i in range(niterations):
    if(show):
        print("Training Network...")
        
    ANN = neuralNetwork(noInputs = 784,noOutputs = 10, hiddenSize=200, learningRatex=0.1)
    # set inputs to be 784 (image pixel size), outputs to be 10 (digits 0-9), size of hidden layer, and learning rate
    ANN.initWeights() # initialize weights
    # mnist - 28x28 image, pixels 0-255

    x = []
    y = []
    for i in range(0,len(train_imgs),10):
        x.append(i)
        ANN.train(train_imgs[i], train_labels_one_hot[i])
        correct = 0
        for j in range(len(test_labels)):
            test_img = test_imgs[j]
            test_label = test_labels[j]
            res, ret = ANN.run(test_img)
            if(res.argmax()==test_label):
                correct+=1
        y.append((correct/(len(test_labels))))
    plt.plot(x,y)
    plt.savefig("trainingimprovement.png")
    plt.close()
    if(show):
        print("Training complete.\n")
        print("Running Network for examples...")
        print("Running on training data (high accuracy - predicted labels should all match actual targets)")

    if(show):
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
        if(i<5 and show):
            print("Prediction = ", res.argmax()," ; Target = ", test_label)
        if(res.argmax()==test_label):
            correct+=1

    acc = 100*correct/len(test_labels)   # accuracy!

    if(show):
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
                x = plt.imread("images\\green_background2.png") # green = win
                plt.title(str(int(test_labels[j])))
            else:
                x = plt.imread("images\\red_background2.png") # red X = lose
                plt.title(str(prediction)+" X, "+str(int(test_labels[j]))+" ✓")
            plt.imshow(x)
            plt.axis('off')
            plt.subplots_adjust(hspace = 2)
        plt.show() # show ploY
        print("Done running.\n\n")

    if(acc>maxacc):
        ANN.saveWeights('weights_test.npz')
        maxacc = acc
        
    print((i+1)," Accuracy: ",acc," % to Max Accuracy: ",maxacc," %")  
