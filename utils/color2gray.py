# This function is used to convert colorful images into grayscale.

import cv2


def color2gray(input_path, output_path):
    name = input_path.split('/')[-1]
    output_path = output_path + 'gray_' + name
    img_array = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize()
    _ = cv2.imwrite(output_path, img_array)
    


input_path = 'C:/Users/rolco/Desktop/old_3.jpg'
output_path = 'C:/Users/rolco/Desktop/'
color2gray(input_path, output_path)
