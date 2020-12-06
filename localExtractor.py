from annoy import AnnoyIndex
import PIL
import numpy as np
from numpy.linalg import norm
import os
from tensorflow.keras.models import Model


class localExtractor:
    def __init__(self, inputModel, folder, saveFile):
        self.model = self.__extract_model_till_layer(inputModel, 10)
        self.folder = folder
        self.file = saveFile

    def __save_features(self, features):
        t = AnnoyIndex(len(features[0]))
        for p in range(len(features)):
            feature = features[p]
            t.add_item(p, feature)

        t.build(40)  # 40 trees
        t.save(self.file)

    def __extract_model_till_layer(self, model, layerNo):

        outputs = model.layers[layerNo].output
        model = Model(inputs=model.inputs, outputs=outputs)
        return model

    def __extract_features(self, file, model):

        a = np.asarray(PIL.Image.open('C://Users//abhig//level8images//'+file))
        a = np.resize(a, (1, 224, 224, 3))
        a = a/255
        features = model.predict(a)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features

    def extract(self):

        all_features = []
        try:
            directory = os.listdir(self.folder)

            for i in range(0, len(directory)):
                features = self.__extract_features(str(i)+".jpg", self.model)
                print("length", len(features))
                all_features.append(features)
                print(i)
                # Length of item vector that will be indexed
                self.__save_features(all_features)

        except:

            self.__save_features(all_features)
