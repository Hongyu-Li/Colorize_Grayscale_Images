from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import ks_2samp


pred_img_path = './preds/preds/'
true_img_path = './trues/trues/'

def ks_test_hist(pred_path,true_path):
    pred_imgs = listdir(pred_path)
    pred_imgs.sort(key=lambda x:int(x[4:-4]))
    true_imgs = listdir(true_path)
    true_imgs.sort(key=lambda x:int(x[4:-4]))
    ks_stat_lst = []
    #p_lst = []
    if len(pred_imgs)!=len(true_imgs):
        print('Test Size is not same!')
    else:
        for i in range(len(pred_imgs)):
            pred_r, pred_g, pred_b = get_flatten_rgb(pred_path+pred_imgs[i])
            true_r, true_g, true_b = get_flatten_rgb(true_path+true_imgs[i])
            ks_stat_r,_ = ks_2samp(pred_r, true_r)
            ks_stat_g,_ = ks_2samp(pred_g, true_g)
            ks_stat_b,_= ks_2samp(pred_b, true_b)
            ks_stat = (ks_stat_r+ks_stat_g+ks_stat_b)/3
            ks_stat_lst.append(ks_stat)
            #p_lst.append([p1,p2,p3])
        return ks_stat_lst

def get_flatten_rgb(img_name):
    img = Image.open(img_name)
    if img.mode=='RGB':
        r,g,b= img.split()
        r = np.asarray(r).flatten()
        g = np.asarray(g).flatten()
        b = np.asarray(b).flatten()
    return r,g,b

import pandas as pd
res_3 = ks_test_hist(pred_img_path,true_img_path)
df = pd.DataFrame({'ours_5':res_3})
df.to_csv('C:/Users/rolco/Desktop/ours_5.csv')
