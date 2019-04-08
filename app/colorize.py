import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from App import app

import base64
import io
import cv2


## Basic header
nav_bar = html.Nav([
    html.Div([
        html.Div([
            html.Button([
                html.Span(className="icon-bar"),
                html.Span(className="icon-bar"),
                html.Span(className="icon-bar")
            ], className="navbar-toggle", type="button", **{'data-toggle': 'collapse'}, **{'data-target': '#myNavbar'}),
            html.A('Image Colorization', href="#myPage", className="navbar-brand")
        ], className="navbar-header"),
        html.Div([
            html.Ul([
                html.Li([html.A('Home', href='/Home')]),
                html.Li([html.A('U Art', href='/Colorize')]),
                html.Li([html.A([
                    html.I(className="fa fa-github")
                ], href='https://github.com/Hongyu-Li/Colorize_Grayscale_Images/')])
            ], className="nav navbar-nav navbar-right")
        ], className="collapse navbar-collapse", id="myNavbar")
    ], className="container")
], className="navbar navbar-default navbar-fixed-top")

## upload image
upload_img = html.Div([
    html.P('Create Something You Want!(png file only)'),
    html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '50%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '0 auto'
                    },
                    multiple=True
                ),
                html.Br(),
                html.Div(id='output-image-upload'),
            ]),  
    ], style={
        'textAlign': 'center',
    })
], className="jumbotron text-center")

def parse_contents(contents):
    content_type, content_string = contents.split(',')      
    decoded = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(decoded)))/255.
    imgsize = img.shape[0:2]
    pred_img = colorization(img,256,256)
    pred_img = cv2.resize(pred_img,imgsize[::-1])
    buff = io.BytesIO()
    pred_img = Image.fromarray(pred_img,'RGB')
    pred_img.save(buff,format='png')
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '
    show_img = html.Div([
        html.Img(src=contents,width="49%", height='70%'),
        html.Img(id=f'img-{id}',
                 src=HTML_IMG_SRC_PARAMETERS + encoded_img, 
                 width="49%",height='70%')
    ])
    return show_img

@app.callback(Output('output-image-upload','children'),
              [Input('upload-image','contents')])

def update_output(contents):
    if contents is not None:
        children = [parse_contents(contents[0])]
        return children

import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

from os import listdir
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import time
from PIL import Image
import numpy as np
import urllib
import cv2
from IPython import display
from tensorflow.python.keras import models
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization,LeakyReLU,Dropout,Add,concatenate

def colorization(img,width,height):
    img = cv2.resize(img, (width, height)) 
    test_image = np.array(img,dtype='float32')
    if len(test_image.shape) == 3:
      test_array = cv2.cvtColor(test_image,cv2.COLOR_RGB2LAB)
      test_array = np.expand_dims(test_array,axis=0)
    else:
      test_array = cv2.cvtColor(test_image,cv2.COLOR_GRAY2RGB)
      test_array = cv2.cvtColor(test_image,cv2.COLOR_RGB2LAB)
    test_l = np.asarray(test_array[:,:,:,0],dtype='float32')
    test_l = test_l/50 -1
    test_l = np.expand_dims(test_l,axis=3)
    test_img_pred = generator(test_l,training=False).numpy()
    test_img_pred[:,:,:,0] = (test_img_pred[:,:,:,0]+1)*50
    test_img_pred[:,:,:,1:] *= 110
    pred_img = cv2.cvtColor(test_img_pred[0,:,:,:],cv2.COLOR_LAB2RGB)
    pred_img = np.clip(pred_img*255.,0,255).astype('uint8')
      
    return pred_img

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    
    # encoder 
    self.encoder_conv1 = Conv2D(filters=64, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_leakyrelu2 = LeakyReLU(0.2)
    self.encoder_conv2 = Conv2D(filters=128, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_bn2 = BatchNormalization()
    self.encoder_leakyrelu3 = LeakyReLU(0.2)
    self.encoder_conv3 = Conv2D(filters=256, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_bn3 = BatchNormalization()
    self.encoder_leakyrelu4 = LeakyReLU(0.2)
    self.encoder_conv4 = Conv2D(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_bn4 = BatchNormalization()
    self.encoder_leakyrelu5 = LeakyReLU(0.2)
    self.encoder_conv5 = Conv2D(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_bn5 = BatchNormalization()
    self.encoder_leakyrelu6 = LeakyReLU(0.2)
    self.encoder_conv6 = Conv2D(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_bn6 = BatchNormalization()
    self.encoder_leakyrelu7 = LeakyReLU(0.2)
    self.encoder_conv7 = Conv2D(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.encoder_bn7 = BatchNormalization()
    self.encoder_leakyrelu8 = LeakyReLU(0.2)
    self.encoder_conv8 = Conv2D(filters=512, kernel_size=(4,4), strides=2, padding="same",activation='relu')
    
    #decoder
    self.decoder_convtrans1 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn1 = BatchNormalization()
    self.decoder_dropout1 = Dropout(0.5)
    self.decoder_add1 = Add()
    self.decoder_convtrans2 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn2 = BatchNormalization()
    self.decoder_dropout2 = Dropout(0.5)
    self.decoder_add2 = Add()
    self.decoder_convtrans3 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn3 = BatchNormalization()
    self.decoder_dropout3 = Dropout(0.5)
    self.decoder_add3 = Add()
    self.decoder_convtrans4 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn4 = BatchNormalization()
    self.decoder_add4 = Add()
    self.decoder_convtrans5 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn5 = BatchNormalization()
    self.decoder_add5 = Add()
    self.decoder_convtrans6 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn6 = BatchNormalization()
    self.decoder_add6 = Add()
    self.decoder_convtrans7 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=2, padding="same")
    self.decoder_bn7 = BatchNormalization()
    self.decoder_add7 = Add()
    self.decoder_convtrans8 = Conv2DTranspose(filters=3, kernel_size=(4,4), strides=2, padding="same",activation='tanh')
    
  def call(self, x,training=True):
    """
    x: input grayscale images---->[N*H*W*1]
    """
    e1 = self.encoder_conv1(x)
    e2 = self.encoder_leakyrelu2(e1)
    e2 = self.encoder_conv2(e2)
    e2 = self.encoder_bn2(e2,training=training)
    e3 = self.encoder_leakyrelu3(e2)
    e3 = self.encoder_conv3(e3)
    e3 = self.encoder_bn3(e3,training=training)
    e4 = self.encoder_leakyrelu4(e3)
    e4 = self.encoder_conv4(e4)
    e4 = self.encoder_bn4(e4,training=training)
    e5 = self.encoder_leakyrelu5(e4)
    e5 = self.encoder_conv5(e5)
    e5 = self.encoder_bn5(e5,training=training)
    e6 = self.encoder_leakyrelu6(e5)
    e6 = self.encoder_conv6(e6)
    e6 = self.encoder_bn6(e6,training=training)
    e7 = self.encoder_leakyrelu7(e6)
    e7 = self.encoder_conv7(e7)
    e7 = self.encoder_bn7(e7,training=training)
    e8 = self.encoder_leakyrelu8(e7)
    e8 = self.encoder_conv8(e8)
    
    d1 = tf.nn.relu(e8)
    d1 = self.decoder_convtrans1(d1)
    d1 = self.decoder_bn1(d1,training=training)
    d1 = self.decoder_dropout1(d1)
    d1 = self.decoder_add1([d1,e7])
    d2 = tf.nn.relu(d1)
    d2 = self.decoder_convtrans2(d2)
    d2 = self.decoder_bn2(d2,training=training)
    d2 = self.decoder_dropout2(d2)
    d2 = self.decoder_add2([d2,e6])
    d3 = tf.nn.relu(d2)
    d3 = self.decoder_convtrans3(d3)
    d3 = self.decoder_bn3(d3,training=training)
    d3 = self.decoder_dropout3(d3)
    d3 = self.decoder_add3([d3,e5])
    d4 = tf.nn.relu(d3)
    d4 = self.decoder_convtrans4(d4)
    d4 = self.decoder_bn4(d4,training=training)
    d4 = self.decoder_add4([d4,e4])
    d5 = tf.nn.relu(d4)
    d5 = self.decoder_convtrans5(d5)
    d5 = self.decoder_bn5(d5,training=training)
    d5 = self.decoder_add5([d5,e3])
    d6 = tf.nn.relu(d5)
    d6 = self.decoder_convtrans6(d6)
    d6 = self.decoder_bn6(d6,training=training)
    d6 = self.decoder_add6([d6,e2])
    d7 = tf.nn.relu(d6)
    d7 = self.decoder_convtrans7(d7)
    d7 = self.decoder_bn7(d7,training=training)
    d7 = self.decoder_add7([d7,e1])
    d8 =tf.nn.relu(d7)
    d8 = self.decoder_convtrans8(d8)
    
    return d8
    
def load_model():
    generator=Generator()
    # Initialize somehow!
    _=generator(tfe.Variable(np.zeros((1,256,256,1),dtype=np.float32)), training=True)
    web='http://www.columbia.edu/~hl3099/csu/generator_epoch60_weight_100.h5' 
    urllib.request.urlretrieve(web,'generator_epoch60_weight_100.h5')
    generator.load_weights('generator_epoch60_weight_100.h5')
    return generator
  
generator=load_model()

layout = html.Div([ 
    nav_bar,
    upload_img
])