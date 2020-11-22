from annoy import AnnoyIndex
import requests
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
import time
import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import random
import time
import math
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D


class GIBSDownloader:

    __init__(self, zoom, date,folder):
        self.zoomLevel=zoom
        self.date=date #yyyy-mm-dd
        self.folder=folder
    
    def download(self):
        
        if(self.zoomLevel==8):
            counter=0
            for i in range(0,160):
                    for j in range(0,320):
                    
                        
                                
                        with open(self.folder+str(counter)+'.jpg', 'wb') as handle:
                                response = requests.get("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"+self.date+"/250m/"+self.zoomLevel+"/"+str(i)+"/"+str(j)+".jpg",stream=True)

                                if not response.ok:
                                    print(response)

                                for block in response.iter_content(1024):
                                    if not block:
                                        break

                                    handle.write(block)
                        print(str(i),str(j))
                                
                        counter+=1
            if(self.zoomLevel==4):
                counter=0
                for i in range(0,10):
                        for j in range(0,20):
                        
                            
                                    
                            with open(self.folder+str(counter)+'.jpg', 'wb') as handle:
                                    response = requests.get("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"+self.date+"/250m/"+self.zoomLevel+"/"+str(i)+"/"+str(j)+".jpg",stream=True)

                                    if not response.ok:
                                        print(response)

                                    for block in response.iter_content(1024):
                                        if not block:
                                            break

                                        handle.write(block)
                            print(str(i),str(j))
                                    
                            counter+=1
