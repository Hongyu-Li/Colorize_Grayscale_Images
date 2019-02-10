from selenium import webdriver  #simulate scroll the website
import urllib.request #save images
import re #regular expression
import time


class Unsplash:
    #初始化构造函数
    def __init__(self):
        self.url='https://unsplash.com/'  
        self.save_path="E:/CSU/毕业论文/数据集/sampling_dataset/Unsplash_2000/"
        self.driver=webdriver.Chrome('C:/Program Files (x86)/Google/Chrome/Application/chromedriver')
       
    def do_scroll(self,times):
        # do scroll the website and get images 
        driver=self.driver
        driver.get(self.url)  # open the website
        #simulate scroll and load images
        for i in range(times):
            print('Scroll '+str(i+1)+' :')
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print('Waitting for '+ str(i+1)+ 'th loading.')
            time.sleep(40)
        html = driver.page_source
        return html

    # save images
    def save_img(self,src,img_name):
        urllib.request.urlretrieve(src, filename=self.save_path + img_name)

    def get_pic(self, html):
        #regular expression so as to get the url of images
        pattern = re.compile('href="https://unsplash.com/photos/[A-Za-z0-9_]*/download\\?force=true"',re.S)
        items = re.findall(pattern, html)
        #download images
        for url in items:
            _,url = url.split('href=')
            url = url.replace('"','')
            name = url.replace('https://unsplash.com/photos/','')
            name = name.split('/')[0] + '.jpg'
            self.save_img(url,name)

    def main(self):
        #get complete html files
        html=self.do_scroll(230)
        print("Start Downloading Images")
        self.get_pic(html)

img=Unsplash()
img.main()



# split train and test
from os import listdir
import random
from PIL import Image

file_name = 'E:/CSU/毕业论文/数据集/Unsplash/'
output_path = 'E:/CSU/毕业论文/数据集/sampling_dataset/Unsplash_2000/'
num_train = 2000
num_test = 200

def get_files(file_name,num_train,num_test,output_path):
    img_lst = listdir(file_name)
    #imgs = np.random.choice(img_lst,num_train+num_test)
    random.shuffle(img_lst)
    for img in img_lst[:2000]:
        temp = Image.open(file_name + img)
        temp.save(output_path+'train/'+img)
    for img in img_lst[2000:2200]:
        temp = Image.open(file_name + img)
        temp.save(output_path+'test/'+img)
        
get_files(file_name,num_train,num_test,output_path)
    
