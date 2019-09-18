import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage import color
from skimage import io
from skimage.measure import block_reduce
from skimage.transform import rescale

blurredness = 2 # controls how blurry gaussian filter is
greyleniency = 16 # how many tones of leniency for considering something absolute white for bounding box, i.e. 3 --> anything >252 is white, <3 is black

# plots + shows image
def showimg(image):
    plt.imshow(image,cmap='gray')
    plt.show()

# crops out white segments of up to the amount permitted by greyleniency
def boundingbox(image):
    # variables showing indexes for the rows/column bounds
    left_bound = 0
    right_bound = len(image)-1
    upper_bound = 0
    lower_bound = len(image[0])-1

    # loops through array to find left, right and upper bounds
    # for LOWER
    # allwhite- boolean value showing if all pixels come across have been white
    allwhite = True
    # for every row
    for i in range(len(image)-1,-1,-1):
        if(allwhite):
            lower_bound=i # as every row gone through is allwhite, the bound is the index value of the current row
        else:
            #breaks from loop
            break
        # iterates through every pixel in row
        for j in range(0,len(image[0])):
            intx = image[i][j]
            if not(intx>=(255-greyleniency)):
                allwhite = False
    # for UPPER
    # allwhite- boolean value showing if all pixels come across have been white
    allwhite = True
    # for every row
    for i in range(0, len(image)):
        if(allwhite):
            upper_bound=i # as every row gone through is all white, the bound is the index value of the current row
        else:
            break
        # iterates through every pixel in row
        for j in range(0, len(image[0])):
            intx = image[i][j]
            if not(intx>=(255-greyleniency)):
                allwhite = False
    # for LEFT
    # allwhite- boolean value showing if all pixels come across have been white
    allwhite = True
    # for every row
    for j in range(0, len(image[0])):
        if(allwhite):
            left_bound=j # as every row gone through is allwhite, the bound is the index value of the current row
        else:
            #breaks from loop
            break
        # iterates through every pixel in row
        for i in range(0, len(image)):
            intx = image[i][j]
            if not(intx>=(255-greyleniency)):
                allwhite = False
    # for RIGHT
    # allwhite- boolean value showing if all pixels come across have been white
    allwhite = True
    # for every row
    for j in range(len(image[0])-1,-1,-1):
        if(allwhite):
            right_bound=j# as every row gone through is all white, the bound is the index value of the current row
        else:
            break
        # iterates through every pixel in row
        for i in range(0, len(image)):
            intx = image[i][j]
            if not(intx>=(255-greyleniency)):
                allwhite = False

    # prints new bounds
    print("Right bound: "+str(right_bound))
    print("Left bound: "+str(left_bound))
    print("Upper bound: "+str(upper_bound))
    print("Lower bound: "+str(lower_bound))
    
    # creates new, smaller array according to the new bounds
    new_array = []
    for i in range(upper_bound, lower_bound+1):
        temp = []
        for j in range(left_bound,right_bound+1):
            temp.append(image[i][j])
        new_array.append(temp)
    return(new_array)

# sets a rectangular image in a square of side length image's largest dimension + padding addp
def insquare(image,addp):
    # find height + width
    height = len(image[0])
    width = len(image)
    # what each value from original image array needs to be offset by to center it in the new square array
    offsetheight = addp
    offsetwidth = addp
    # take the larger value
    if(height>width):
        sqlength = height+2*addp
        offsetwidth = int((height-width)/2+0.5)+addp
    else:
        sqlength = width+2*addp
        offsetheight += int((width-height)/2+0.5)+addp
    # initialize it to 255 (white)
    new_array = np.full((sqlength,sqlength),255)
    # add values from image array into new array offset by the offset values
    for i in range(0, len(image)):
        for j in range(0,len(image[0])):
            new_array[i+offsetwidth][j+offsetheight] = image[i][j]
    # returns new centered square image array
    return(new_array)

# path on computer where image is stored
data_path = "C:\\Users\\creag\\OneDrive\\compsci-proj\\test_image.png"

# converts to greyscale not rgb photo
test_image = color.rgb2gray(io.imread(data_path))
print("Greyscale image loaded.")
showimg(test_image)

# blurs with gaussian filter
blurred_image = gaussian_filter(test_image,sigma=blurredness)
print("Gaussian filter applied.")
showimg(blurred_image)

# cuts whitespace to put image in bounding box
bounded_image = boundingbox(blurred_image)
print("Image cropped. ")
showimg(bounded_image)

# centers bounding box inside square
squared_image = insquare(bounded_image,0)
print("Image squared.")

# rescales image to 20x20 pixel box, 8-bit resolution
rescaling_factor = 20/len(squared_image)
# rescaling factor is what the length of arrays are multiplied by to make smaller
rescaled_image = rescale(np.asarray(bounded_image), rescaling_factor, anti_aliasing=False,multichannel=False)
print("Image rescaled.")
showimg(rescaled_image)

# centers new image in 28x28 box
# already in 20x20 pixel box so the added factor for padding is 4 (1/2 of 8) 
padded_image = insquare(rescaled_image, 4)
print("Image centered + padded.")
showimg(padded_image)

