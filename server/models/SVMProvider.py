import pickle

import numpy as np
from config.ConfigLoader import ConfigLoader
from sklearn.svm import SVC


class SVMProvider():
    
    model = None
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.model = None
        self.model_config = config    

    def load(self):
        if (self.model is not None):
            print("Model already loaded")
            return

        # Load SVM model from pickle file
        self.model = pickle.load(open(self.model_path, 'rb'))
        self.model.set_params(C=self.model_config['C'], kernel=self.model_config['kernel'], gamma=self.model_config['gamma'])
    
    def predict(self, features):
        pred = self.model.predict(features)
        conf = None
        return float(pred), conf
