# USES MNIST DATASET
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets
import random

print("STARTING...")

# digits - 8x8 image, pixels 0-16
# mnist - 28x28 image, pixels 0-255
data_path = "C:\\Users\\creag\\Downloads\\"
train_data = np.empty([60000,785])
row = 0
## FORMAT:: comes as comma seperated text file
#w/ first number being number it's supposed to be, then 784 pixel values 0-255
# so uploaded both test/train files into arrays of 785, first being number and rest being image
for line in open(data_path + "mnist_train.csv"):
    train_data[row] = np.fromstring(line, sep=",")
    row += 1
test_data = np.empty([10000,785])
row = 0
for line in open(data_path + "mnist_test.csv"):
    test_data[row] = np.fromstring(line, sep=",")
    row += 1

print("DATA LOADED...")
 # what's in the skikit learn digits
# type of the columns?
print (type(train_data))
print (type(train_data))
# number of dimensions, size of each
print(train_data[0])
print(train_data.shape)
plt.show()
#plt.imshow(digits.images[100],cmap='binary')
# print image
plt.show()

random.shuffle(train_data)
random.shuffle(test_data)
train_size = 20000
test_size =  3000
#making the y of the training set: the digits of the image, the 0th index number froem each image
y_train = np.empty(train_size)
for i in range(train_size):
    y_train[i] = train_data[i][0]
    
# making the x of the training set= the actual images
x_train = np.empty([train_size,784])
for i in range(train_size):
    x_train[i] = train_data[i][1:]

#making the y of the test set: the digits of the image, the 0th index number froem each image
y_test = np.empty(test_size)
for i in range(test_size):
    y_test[i] = test_data[i][0]
    
# making the x of the test set= the actual images
x_test = np.empty([test_size,784])
for i in range(test_size):
    x_test[i] = test_data[i][1:]

def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(28,28))
    for j in range(nplots):
        plt.subplot(4,4,j+1) # divides window into smaller plots
        plt.imshow(np.reshape(x_train[i+j],(28,28)), cmap='binary')
        plt.title(y_train[i+j])
        plt.axis('off')
    plt.show()

plot_multi(random.randint(0,(train_size-16)))
# make sure you flatten the 8x8 arrays not the 1797 images x image width or something else etc
# split into testing + training set
# so that you never get a funciton that only works on testing set 100% of time,
# so unseen testing set is important so can see how close function is to actual function

from sklearn.neural_network import MLPClassifier

#MLP Classifier = classifies input into categories = neural network
#how many neurons in hidden layer = 15, sigmoid curve = 'logistic' activation, alpha = stops overfitting by penalizing weights
#'sgd' = stochastic type of gradient descent (go through examples 1 by 1- example+predict output, compare, etc), not batch (for all examples predict output + average), minibatch = stochastic with small batches
# tolerance = how much error must decrease in iteration
# learning rate is steps for gradient descent, small so accurate, verbose = prints messages
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)

# into fit- input, output
# predicts outputs/compares- training
mlp.fit(x_train,y_train)

# plots a few (90)examples of our predictions/dataset
def plot_predicts(i, predictions):
    '''Plots 16 digits, starting with digit i'''
    nplots = 90
    fig = plt.figure()
    fig.tight_layout()
    for j in range(nplots):
        plt.subplot(10,9,j+1) # divides window into smaller plots
        plt.imshow(np.reshape(x_test[i+j],(28,28)), cmap='binary')
        if(predictions[i+j]==y_test[i+j]):
            x = plt.imread("C:\\Users\\creag\\Downloads\\green_background2.png")
            plt.title(str(y_test[i+j]))
        else:
            x = plt.imread("C:\\Users\\creag\\Downloads\\red_background2.png")
            plt.title(str(predictions[i+j])+" : "+str(y_test[i+j]))
        plt.imshow(x)
        plt.axis('off')
        plt.subplots_adjust(hspace = .2)
    plt.show()

# predictions- puts new data into neural network predict function (already trained)
predictions = mlp.predict(x_test)
plot_predicts(random.randint(0,(test_size-90)), predictions)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
# gives accuracy score of our network's predictions - 90 percent

predictions2 = mlp.predict(x_train)
print(accuracy_score(y_train,predictions2))

