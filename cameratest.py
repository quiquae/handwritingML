from gpiozero import Button
from picamera import PiCamera
import matplotlib.pyplot as mp
import matplotlib.image as mi

camera = PiCamera()

def takephoto():
    camera.capture('/home/pi/Pictures/photos1/testimage.jpg')
    mp.imshow(mi.imread('/home/pi/Pictures/photos1/testimage.jpg'))
    mp.show()
    print('photoo taken')
    
button = Button(4)
while(True):
    button.when_pressed = takephoto
