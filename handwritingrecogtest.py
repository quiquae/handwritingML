import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets

# digits - 8x8 image, pixels 0-16
# mnist - 28x28 image, pixels ??
digits = datasets.load_digits()

print(dir(digits)) # what's in the skikit learn digits
# type of the columns?
print (type(digits.images))
print (type(digits.target))
# number of dimensions, size of each
print(digits.images.shape)
# 1797 images that are 8x8

#plt.imshow(digits.images[100],cmap='binary')
# print image
plt.show()


def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1) # divides window into smaller plots
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()

#plot_multi(40)

# flattens 'matrix' 8x8 2D array to 64 1d array 'vector'
y = digits.target
x = digits.images.reshape((len(digits.images), -1))
# make sure you flatten the 8x8 arrays not the 1797 images x image width or something else etc
x.shape

# split into testing + training set
# so that you never get a funciton that only works on testing set 100% of time,
# so unseen testing set is important so can see how close function is to actual function
# ratio is from 60:40 to 80:20
x_train = x[:1200]
y_train = y[:1200]
#training is before 1200, testing is after
x_test = x[1200:]
y_test = y[1200:]


from sklearn.neural_network import MLPClassifier

#MLP Classifier = classifies input into categories = neural network
#how many neurons in hidden layer = 15, sigmoid curve = 'logistic' activation, alpha = stops overfitting by penalizing weights
#'sgd' = stochastic type of gradient descent (go through examples 1 by 1- example+predict output, compare, etc), not batch (for all examples predict output + average), minibatch = stochastic with small batches
# tolerance = how much error must decrease in iteration
# learning rate is steps for gradient descent, small so accurate, verbose = prints messages
mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)

# into fit- input, output
# predicts outputs/compares- training
mlp.fit(x_train,y_train)

def plot_multi(i, predictions):
    '''Plots 16 digits, starting with digit i'''
    nplots = 25
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(5,5,j+1) # divides window into smaller plots
        plt.imshow(digits.images[1200+i+j], cmap='binary')
        plt.title(str(predictions[i+j])+" : "+str(y_test[i+j]))
        plt.axis('off')
    plt.show()


# predictions- puts new data into neural network predict function (already trained)
predictions = mlp.predict(x_test)
plot_multi(105, predictions)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
# gives accuracy score of our network's predictions - 90 percent


