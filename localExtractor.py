from annoy import AnnoyIndex
import requests
from io import BytesIO
import cv2
import PIL
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
import wget


class localExtractor:
    def __init__(self, inputModel, folder, saveFile):
        self.model=self.extract_features(inputModel,10)
        self.folder=folder
        self.file=saveFile

    def extract_model_till_layer(self,model, layerNo):
        
        outputs = model.layers[layerNo].output
        model = Model(inputs=model.inputs, outputs=outputs)
        return model

    def extract_features(self,file, model):
                
        a = np.asarray(PIL.Image.open('C://Users//abhig//level8images//'+file))
        a=np.resize(a,(1,224,224,3))
        a=a/255       
        #preprocessed_img = preprocess_input(a)
        features = model.predict(a)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features

    def extract(self):
                            
        features=[]
        try:
            
        
            directory=os.listdir(self.folder)
        
            for i in range(0,len(directory)):
                feats=self.extract_features(str(i)+".jpg",self.model)
                print("length",len(feats))
                features.append(feats)
                print(i)
                counter+=1
        
                


                # Length of item vector that will be indexed
                t=AnnoyIndex(len(features[0]))
                for p in range(len(features)):
                    feature = features[p]
                    t.add_item(p, feature)

                t.build(40)  # 40 trees
                t.save(self.file)

        except:
            
            t=AnnoyIndex(features[0])
            for p in range(len(features)):
                feature = features[p]
                t.add_item(p, feature)

            t.build(40)  # 40 trees
            t.save(self.file)
