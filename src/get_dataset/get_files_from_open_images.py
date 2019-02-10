import pandas as pd
from six.moves import urllib
import random

train_data = pd.read_csv('train-images-boxable.csv',sep=',')
test_data = pd.read_csv('test-images.csv',sep=',')
train_url = train_data.iloc[:,0].values + '\n' + train_data.iloc[:,1].values
test_url = test_data.iloc[:,0].values + '\n' + test_data.iloc[:,1].values


def get_files(url,num_images,output_path):
    count = 0
    random.shuffle(url)
    for line in url: 
        id_img, img_url = line.split('\n')
        name = output_path + id_img + '.jpg'
        try:
            urllib.request.urlretrieve(img_url,name)
        except:
            continue
        count += 1
        if count >= num_images:
            break
        
get_files(train_url,2000,'E:/CSU/毕业论文/数据集/sampling_dataset/Open_2000/train/')
get_files(test_url,200,'E:/CSU/毕业论文/数据集/sampling_dataset/Open_2000/test/')