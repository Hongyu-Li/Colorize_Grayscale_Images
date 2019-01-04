import numpy as np
import cv2

def color2gray(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    name = path.split('_')[0]
    cv2.imwrite(name+'_gray.jpg',img)



