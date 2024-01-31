from exp.FFNet import FFNet
from config.ConfigLoader import ConfigLoader

class FFNetProvider():

    model = None
    def __init__(self, model_path, model_features_size, config):
        self.model_path = model_path
        self.model_features_size = model_features_size
        self.model = None
        self.model_config = config

    def load(self):
        if (self.model is not None):
            print("Model already loaded")
            return        
        
        self.model = FFNet(self.model_config['input'], self.model_config['hidden'], self.model_config['output'], self.model_config["activations"], self.model_config["dropout_positions"], self.model_config["dropout_probs"])
        self.model.load(self.model_path)
    
    def predict(self, features):
        return self.model.predict(features)
