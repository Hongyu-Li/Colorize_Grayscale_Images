from os import listdir
import random
from PIL import Image

file_name = 'E:/CSU/毕业论文/数据集/COCO/unlabeled2017/'
output_path = 'E:/CSU/毕业论文/数据集/sampling_dataset/COCO_2000/'
num_train = 2000
num_test = 200

def get_files(file_name,num_train,num_test,output_path):
    img_lst = listdir(file_name)
    #imgs = np.random.choice(img_lst,num_train+num_test)
    random.shuffle(img_lst)
    img_iter = iter(img_lst)
    #img_train = imgs[:num_train]
    #img_test = imgs[num_train:]
    control = 0
    while control < num_train:
        img = next(img_iter)
        temp = Image.open(file_name + img)
        if temp:
            temp.save(output_path+'train/'+img)
            control += 1
    
    control = 0
    while control < num_test:
        img = next(img_iter)
        temp = Image.open(file_name + img)
        if temp:
            temp.save(output_path+'test/'+img)
            control += 1
        
get_files(file_name,num_train,num_test,output_path)