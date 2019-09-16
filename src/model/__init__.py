from keras.applications.nasnet import NASNetLarge
from keras.models import load_model
import os
import numpy as np

FRAME_INTERVAL = 25

class Model:
    def __init__ (self):
        if not os.path.isfile("model.h5"):
            print("no model file")
            exit(1)
        print("loading model")
        self.classifier = load_model('model.h5')
        self.nas = NASNetLarge(input_shape=None, include_top=True,
                        weights='imagenet', input_tensor=None, pooling=None, classes=1000)
        self.nas.layers.pop()
        self.nas.outputs = [self.nas.layers[-1].output]
        self.nas.layers[-1].outbound_nodes = []

    def predict(self, segment):
        features = self.nas.predict(np.array(segment))
        result = self.classifier.predict(
            np.array(features).reshape(1, FRAME_INTERVAL, 4032))
        print(result)
        del segment[:]
        return np.argmax(result[0], axis=0)
