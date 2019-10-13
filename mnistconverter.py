# every single thing i have to import for this image edit code (my goD!)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import scipy.ndimage as nd
from skimage import color, io, img_as_ubyte
from skimage.measure import block_reduce
from skimage.transform import rescale

blurredness = 2 # controls how blurry gaussian filter is
greyleniency = 16 # how many tones of leniency for considering something absolute white for bounding box, i.e. 3 --> anything >252 is white, <3 is black

#-----------------------------------------------------------------------
#----------------------------- BASIC FUNCTIONS------------------------
#-----------------------------------------------------------------------

# plots + shows image
def showimg(image):
    plt.imshow(image,cmap='gray')
    plt.show()

# crops out white segments of up to the amount permitted by greyleniency
# theres probably a library with a special function for this but i decided to code myself :)
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
            if not(intx>=(0+greyleniency)):
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
            if not(intx>=(0+greyleniency)):
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
            if not(intx>=(0+greyleniency)):
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
            if not(intx>=(0+greyleniency)):
                allwhite = False

    # prints new bounds
##    print("Right bound: "+str(right_bound))
##    print("Left bound: "+str(left_bound))
##    print("Upper bound: "+str(upper_bound))
##    print("Lower bound: "+str(lower_bound))
    
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
##    print("width: ", width)
##    print("height: ", height)
    # what each value from original image array needs to be offset by to center it in the new square array
    offsetheight = addp
    offsetwidth = addp
    # take the larger value
    if(height>width):
        sqlength = height+2*addp
        offsetwidth = int((height-width)/2+0.5)
    elif(height<width): #CHANGED THIS ELSE->ELIF (EXCLUDE EQUALITY i.e. square CASE)
        sqlength = width+2*addp
        offsetheight += int((width-height)/2+0.5) #removed "+addp" as using += and already set offsetheight as addp!! REMEMBER
##        print(offsetheight)
##        print("ENTERRED HERE1")
    else:
        sqlength = width+2*addp
##        print("ENTERRED HERE2")

    # initialize it to 255 (white)
    new_array = np.full((sqlength,sqlength),0)
    # add values from image array into new array offset by the offset values
    for i in range(0, len(image)):
        for j in range(0,len(image[0])):
            new_array[i+offsetwidth][j+offsetheight] = image[i][j]
    # returns new centered square image array
    return(new_array)

#---------------------------------------------------------------------
#-------------------- COMPLEX/AGGREGATE FUNCTIONS---------------------
#---------------------------------------------------------------------

# best shift
def getbestshift(image):
    # finds center of mass
    cy,cx = nd.measurements.center_of_mass(image)
    rows,cols = image.shape
    # offset between center of mass of digit and center of image
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

# shifts image according to shift paramenters
def shift(image, shiftx, shifty):
    r,c = image.shape
    M = np.float32([[1,0,shiftx],[0,1,shifty]])
    shifted = cv2.warpAffine(image, M, (c,r))
    return(shifted)

# open image in grayscale
def openimage(data_path):
    gray = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
    return(gray)

def gaussian(image, blur):
    # gaussian blur- yay!
    return(nd.gaussian_filter(image, blur))

def pixelsquare(image):
    # fits bounding box to image
    bounded_image = boundingbox(image)
    # puts inside square
    squared_image = insquare(bounded_image,0)
    # pixelates/rescales it to 24x24
    rescaling_factor = 20/len(squared_image)
    rescaled_image = rescale(np.asarray(bounded_image), rescaling_factor, anti_aliasing=False,multichannel=False)
    rescaled_image = img_as_ubyte(rescaled_image)
    # pads 24x24 square inside 28x28 whitespace
    padded_image = insquare(rescaled_image, 4)
    return(padded_image)

# opens image and converts it to MNIST format!
# datapath = where image is stored, blur = gaussian blur/sigma, showimage = whether image is plotted during function
def convert(datapath, blur, showimage):
    if(showimage):
        showimg(mpimg.imread(datapath))
    # loads orignally rgb image to grey + adds gaussian blur
    image = openimage(datapath)
    image = cv2.resize(255-image, (28, 28))
    # resize it 
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #image = gaussian(image, blur)
    #showimg(image)
    # puts inside pixelated square
    fac = 0.99 / 255
    image = np.asfarray(image) * fac + 0.01
    image = pixelsquare(image)
    sx, sy = getbestshift(image)
    shifted = shift(image, sx, sy)
    image = shifted
    # shows depending on the value of bool showimage
    if(showimage):
        showimg(image)
    return(image)

##def splitx(image):


#----------------------------------------------------------------------------------------
#---------------------------- USAGE EXAMPLE ---------------------------------------------
#----------------------------------------------------------------------------------------

### path on computer where image is stored
##data_path = "C:\\Users\\creag\\OneDrive\\compsci-proj\\actual_image2.jpeg"
##
### loads photo + converts to greyscale not rgb photo!!!
##test_image = cv2.cvtColor(io.imread(data_path),cv2.COLOR_BGR2GRAY)
##print("Greyscale image loaded.")
##showimg(test_image)
##
### converts to binary
##print("Image converted to binary.")
##
##    # uses weird threshold thing- DO NOT ALTER OR WILL NOT WORK:
##    #cv2 threshold for the 0-255 greyscale value to turn to black/white depending on value "minimizing intra-class intensity variance"  (otsu)
##z,binary_image = cv2.threshold(test_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##showimg(binary_image)
##
### blurs with gaussian filter
##blurred_image = gaussian_filter(binary_image,sigma=blurredness)
##print("Gaussian filter applied.")
##showimg(blurred_image)
##
### cuts whitespace to put image in bounding box
##bounded_image = boundingbox(blurred_image)
##print("Image cropped. ")
##showimg(bounded_image)
##
### centers bounding box inside square
##squared_image = insquare(bounded_image,0)
##print("Image squared.")
##showimg(squared_image)
##
### rescales image to 20x20 pixel box, 8-bit resolution
##rescaling_factor = 20/len(squared_image)
### rescaling factor is what the length of arrays are multiplied by to make smaller
##rescaled_image = rescale(np.asarray(bounded_image), rescaling_factor, anti_aliasing=False,multichannel=False)
##print("Image rescaled.")
##showimg(rescaled_image)
### converts from float to ubyte (integer 0-255) image
##rescaled_image = img_as_ubyte(rescaled_image)
##
### centers new image in 28x28 box
### already in 20x20 pixel box so the added factor for padding is 4 (1/2 of 8) 
##padded_image = insquare(rescaled_image, 4)
##print("Image centered + padded.")
##showimg(padded_image)
##
##
