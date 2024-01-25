import json

class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None

    def load(self):
        print("Loading config", self.config_file)
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
            # Count the number of model features
            num_features = len(self.config['ffnn']['model_features'])
            if (self.config["ffnn"]["model_config"]["input"] != num_features):
                print("Number of model features and input size do not match: Input Size is ")
            # Update the input attribute
            self.config['ffnn']['model_config']['input'] = num_features

             # Write the updated config back to the JSON file
            with open(self.config_file, 'w') as file:
                json.dump(self.config, file, indent=4)
    
    def get_config(self):
        if self.config is None:
            raise Exception("Config not loaded")
        return self.config

if __name__ == "__main__":
    config_loader = ConfigLoader("./config/config.json")
    config_loader.load()
    config = config_loader.get_config()
    print(config)