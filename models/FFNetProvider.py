from exp.FFNet import FFNet
import torch

class FFNetProvider():

    model = None
    def __init__(self, model_path, model_features_size):
        self.model_path = model_path
        self.model_features_size = model_features_size
        self.model = None

    def load(self):
        if (self.model is not None):
            print("Model already loaded")
            return
        config = {
            "input": self.model_features_size,
            "output": 1,
            "hidden": [64, 80, 126, 80, 64, 32],
            "activations": ["ReLU", "ReLU", "ReLU", "ReLU","ReLU", "ReLU", "Sigmoid"],
            "dropout_positions": [],
            "dropout_probs": [],
        }
        
        self.model = FFNet(config['input'], config['hidden'], config['output'], config["activations"], config["dropout_positions"], config["dropout_probs"])
        self.model.load(self.model_path)
    
    def predict(self, features):
        return self.model.predict(features)
