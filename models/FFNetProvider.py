from exp.FFNet import FFNet
from config.ConfigLoader import ConfigLoader

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
        config_loader = ConfigLoader("./config/config.json")
        config_loader.load()
        model_config = config_loader.get_config()["ffnn"]["model_config"]
        
        
        self.model = FFNet(model_config['input'], model_config['hidden'], model_config['output'], model_config["activations"], model_config["dropout_positions"], model_config["dropout_probs"])
        self.model.load(self.model_path)
    
    def predict(self, features):
        return self.model.predict(features)
