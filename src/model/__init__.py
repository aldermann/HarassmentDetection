from keras.applications.nasnet import NASNetLarge
from keras.models import load_model
import os
import numpy as np
from video import FRAME_INTERVAL

class Model:
    def __init__ (self, debug=False):
        self.debug = debug
        if debug:
            return
        model_path = "{}/model/model.h5".format(os.getcwd())
        if not os.path.isfile(model_path):
            print("no model file")
            exit(1)
        print("loading model")
        self.classifier = load_model(model_path)
        self.nas = NASNetLarge(input_shape=None, include_top=True,
                        weights='imagenet', input_tensor=None, pooling=None, classes=1000)
        self.nas.layers.pop()
        self.nas.outputs = [self.nas.layers[-1].output]
        self.nas.layers[-1].outbound_nodes = []

    def predict(self, segment):
        if self.debug:
            return 1
        features = self.nas.predict(np.array(segment))
        result = self.classifier.predict(
            np.array(features).reshape(1, FRAME_INTERVAL, 4032))
        return np.argmax(result[0], axis=0)
