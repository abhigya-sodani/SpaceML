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
import wget
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

class GIBSExtractor():
    def __init__(self,inputModel,date,inputZoom,saveFile):
        self.givenModel=inputModel
        self.model = self.extract_model_till_layer(tf.keras.models.load_model(self.givenModel),10) #replace with whichever model you wish to featurize with
        self.date=date #yyyy-mm-dd
        self.zoom=inputZoom
        self.file=saveFile

    def extract_model_till_layer(self, model, layerNo):
  
      outputs = model.layers[layerNo].output
      model = Model(inputs=model.inputs, outputs=outputs)
      return model
    def extract_features(self, url, model):
        
       
        response = wget.download(url)
        a = cv2.imread(response)
        os.remove(response)
        a=np.resize(a,(1,224,224,3))
        a=a/255      
        features = model.predict(a)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features

    def extract(self):
        features=[]
        try:
            
        
        
            if(self.zoom==4):

                for i in range(0,10):
                    for j in range(0,20):
                        feats=self.extract_features("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"+self.date+"/250m/"+self.zoom+"/"+str(i)+"/"+str(j)+".jpg", self.model)
                        print("length",len(feats))
                        features.append(feats)
                        print(str(i)+" "+str(j))
                        
                    time.sleep(30)
            
            if(self.zoom==8):

                for i in range(0,160):
                    for j in range(0,320):
                        feats=self.extract_features("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"+self.date+"/250m/"+self.zoom+"/"+str(i)+"/"+str(j)+".jpg", self.model)
                        print("length",len(feats))
                        features.append(feats)
                        print(str(i)+" "+str(j))
                        
                    time.sleep(30)



            # Length of item vector that will be indexed
            t=AnnoyIndex(len(features[0]))
            for p in range(len(features)):
                feature = features[p]
                t.add_item(p, feature)

            t.build(40)  # 40 trees
            t.save(self.file)

        except:
            print("Error occured, indexing the current features")
            t=AnnoyIndex(features[0])
            for p in range(len(features)):
                feature = features[p]
                t.add_item(p, feature)

            t.build(40)  # 40 trees
            t.save(self.file)

            
