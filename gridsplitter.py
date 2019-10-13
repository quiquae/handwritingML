from mnistconverter import convert
from cv2 import imread
from PIL import Image

red_threshold = 50

def findgrid(grid):
    hlinebounds = [] # boundaries for the red horizontal lines (Start x, End x ) for each line
    vlinebounds = [] # boundaries for the red vertical lines (Start x, End x ) for each line
    for i in range(1,len(grid)):
        hlinecounter = 0
        for j in range(1,len(grid[0])):
            nowR = grid[i][j][2]
            thenR = grid[i][j-1][2]
            if(nowR>red_threshold and thenR<red_threshold):
                print("RED START")
                if(len(hlinebounds)==hlinecounter):
                    hlinecounter +=1
                    hlinebounds.append([j,j+1])
                elif(j<(hlinebounds[hlinecounter][0])):
                    hlinecounter +=1
                    hlinebounds[hlinecounter][0] = j
            elif(nowR<red_threshold and thenR>red_threshold):
                print("RED END")
                print(j)
                print(hlinebounds)
                print(hlinecounter)
                if(j>(hlinebounds[hlinecounter-1][1])):
                    hlinecounter+=1
                    hlinebounds[hlinecounter-2][1] = j
    for j in range(1,len(grid)):
        vlinecounter = 0
        for i in range(1,len(grid[0])):
            nowR = grid[i][j][2]
            thenR = grid[i][j-1][2]
            if(nowR>red_threshold and thenR<red_threshold):
                if(len(vlinebounds)==vlinecounter):
                    vlinecounter +=1
                    vlinebounds.append([j,j+1])
                    
                elif(j<vlinebounds[vlinecounter][0]):
                    vlinecounter +=1
                    vlinebounds[vlinecounter][0] = j
            elif(nowR<red_threshold and thenR>red_threshold):
                if(j>hlinebounds[vlinecounter-1][1]):
                    vlinecounter +=1
                    vlinebounds[vlinecounter-2][1] = j
    return(hlinebounds, vlinebounds)
            
                
def splitgrid(image, hbounds, vbounds):
    new_datapath = "data/grid_image"
    for i in range(len(vbounds)-1):
        new_image = []
        for y in range(vbounds[i][1], vbounds[i+1][0]):
            new_image.append(image[y][hbounds[i][1]:hbounds[i+1][0]])
        im = Image.fromarray(new_image)
        im.save((datapath+str(i)))


def gridconvert(datapath, blur, showimage):
    grid = imread(datapath)
    h, v = findgrid(grid)
    splitgrid(grid, h, v)
    datapaths = []
    converted = []
    for image in datapaths:
        converted.append(convert(image, blur, showimage))
    return converted

x = gridconvert("data/redgridtest.jpg",4,False)
