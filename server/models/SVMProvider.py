from config.ConfigLoader import ConfigLoader
import pickle

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
    
    def predict(self, features):
        # TODO: @Pavel: Implement this (prediction and confidence should be returned)
        pred = None
        conf = None
        return float(pred), conf
