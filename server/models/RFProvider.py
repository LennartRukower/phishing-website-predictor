from config.ConfigLoader import ConfigLoader
import pickle

class RFProvider():
    
    model = None
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.model = None
        self.model_config = config

    def load(self):
        if (self.model is not None):
            print("Model already loaded")
            return

        # Load RF model from pickle file
        self.model = pickle.load(open(self.model_path, 'rb'))
    
    def predict(self, features):
        pred = self.model.predict(features)[0]
        class_probabilities = self.model.predict_proba(features)
        conf = class_probabilities[0][pred]
        return float(pred), conf
