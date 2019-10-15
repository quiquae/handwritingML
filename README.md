# handwritingML
Neural network project about reading/classifying handwritten digits, uses MNIST data to model neural net, then can predict digits from non-MNIST normal camera photos

Code complete & functional with ~93% accuracy on MNIST images, ~85% accuracy rating on cropped iPhone photos I've tested.

What's in this repository?
- neuralnet.py as class to create basic neural net
- train.py and test.py (do what they say) train & test neural net on MNIST images
- mnistconverter.py as converter to convert normal photos to MNIST-like input
- WIP!!! gridsplitter.py is me trying to split digits from photo with multiple digits in red grid
- data folder contains lots of test images (actual_image = cropped iPhone photo, test_image = variety of digital images)
- images folder contains images used in programs (colored backgrounds for visualisation, redgrid as test for gridsplitter.py)
- weights folder contains saved weights from neural net program

** IMPORTANT IF USING THIS PROGRAM ** 
Must have mnist_test and mnist_train CSV files of MNIST data for training program to work.
Can either convert from the original website: http://yann.lecun.com/exdb/mnist/, 
or download already converted CSV files from the variety of sources in which they are available, ex. https://www.kaggle.com/oddrationale/mnist-in-csv

(no-one looks at this but might as well have a nice~ README)
