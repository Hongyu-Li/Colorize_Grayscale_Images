from os import listdir
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from PIL import Image
import cv2

height = 224
width  = 224




def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
      if image[::-1][:4] == '.jpg'[::-1]:
          img = Image.open(path + image)
          img = img.resize((width, height))
          img_array = np.asarray(img,dtype='float32')/255.
          if len(img_array.shape) == 3:
              img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
              loadedImages.append(img_array)
    loadedImages = np.asarray(loadedImages,dtype='float32')
    return np.mean(loadedImages, tuple(range(3)),dtype='float32')
  
path = "E:/CSU/毕业论文/数据集/sampling_dataset/train/"
means = loadImages(path)
