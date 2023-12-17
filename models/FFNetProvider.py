from exp.FFNet import FFNet

class FFNetProvider():
    def __init__(self, model_path, model_features_size):
        self.model_path = model_path
        self.model_features_size = model_features_size
        self.model = None

    def load(self):
        config = {
                "input": len(self.model_features_size),
                "output": 2,
                "hidden": [len(self.model_features_size),  48],
                "activations": ["ReLU", "Sigmoid"]
            }
        
        self.model = FFNet(config['input'], config['hidden'], config['output'], config["activations"])
        self.model.load(self.model_path)
    
    def predict(self, features):
        return self.model.predict(features)

