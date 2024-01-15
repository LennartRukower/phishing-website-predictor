from config.ConfigLoader import ConfigLoader
import pickle

class RFProvider():
    
    model = None
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load(self):
        if (self.model is not None):
            print("Model already loaded")
            return
        config_loader = ConfigLoader("../config/config.json")
        config_loader.load()
        config = config_loader.get_config()

        # Load RF model from pickle file
        self.model = pickle.load(open(self.model_path, 'rb'))
    
    def predict(self, features):
        return self.model.predict(features)
