#go to settings (bottom left cog)
#find Python > Linting: Pylint Args (search pylint args)
#add:
# --generate-members
# --extension-pkg-whitelist=cv2

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
data_path = "C:\\Users\\creag\\Downloads\\"
# test image I took from my phone of my own handwriting!! for later
image = convert("C:\\Users\\creag\\OneDrive\\compsci-proj\\actual_image.jpeg",4,True)

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
plt.show() # show ploY

print("Done running.")

ANN.saveWeights('weights_test.npz')
