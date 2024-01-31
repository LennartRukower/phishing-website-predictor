from flask import Flask, jsonify
from flask import request
from services.WebCrawler import WebCrawler
from models.FFNetProvider import FFNetProvider
from services.Preprocessor import Preprocessor
from services.Extractor import Extractor
from config.ConfigLoader import ConfigLoader
from models.RFProvider import RFProvider
import json
from statistics import mode 

app = Flask(__name__)

# Adds CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    return response

models = [
    {
        "name": "ffnn",
        "description": "Use a Feed Forward Neural Network to classify the url",
        "info": None,
        "stats": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    },
    {
        "name": "rf",
        "description": "Use a Random Forest to classify the url",
        "info": None,
        "stats": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    }]

@app.route("/models")
def get_models():
    # Load model version from config
    config_loader = ConfigLoader("config/config.json")
    config_loader.load()
    config = config_loader.get_config()

    for model in models:
        model_version = config[model["name"]]["model_version"]
        model_folder_path = f'exp/models/{model["name"]}/{model_version}'
        # Load stats from info file
        with open(f"{model_folder_path}/info.json", "r") as file:
            info = json.load(file)
            # Round stats to 2 decimals
            model["stats"]["accuracy"] = round(info["accuracy"], 2)
            model["stats"]["precision"] = round(info["precision"], 2)
            model["stats"]["recall"] = round(info["recall"], 2)
            model["stats"]["f1"] = round(info["f1"], 2)
        model["info"] = config[model["name"]]["model_config"]
        
    return jsonify(models), 200

@app.route("/voting", methods=["GET"])
def get_voting_methods():
    voting_methods = [
        {
            "name": "majority",
            "description": "Use the most common prediction",
        },
        {
            "name": "max-confidence",
            "description": "Use the prediction with the highest confidence",
        },
        {
            "name": "average",
            "description": "Use the average of all predictions",
        }
    ]
    return jsonify(voting_methods), 200

@app.route("/predict", methods=["POST"])
def predict_url():
    try:
        # Get the url and model from the request body
        url = request.json.get("url")
        model = request.json.get("model")
        debug = request.args.get("debug")

        # Check if the url and model are valid
        if url is None:
            return jsonify({"error": "URL not provided"}), 400
        if model is None:
            # Use default model
            model = "ffnn"
        if model not in ["ffnn", "rf"]:
            return jsonify({"error": "Model not found"}), 400

        # Crawl the html code    
        cralwer = WebCrawler()
        html = cralwer.crawl(url)
        if html.startswith("Error"):
            return jsonify({"error": html}), 400
        
        # Extract features
        extractor = Extractor()
        extracted_features = extractor.extract_features(url, html)

        # Load config
        config_loader = ConfigLoader("config/config.json")
        config_loader.load()
        config = config_loader.get_config()
        model_features = config[model]["model_features"]
        model_version = config[model]["model_version"]
        model_folder_path = f'exp/models/{model}/{model_version}'

        # Check if extracted features match the model features
        if len(extracted_features) != len(model_features):
            return jsonify({"error": "Extracted features do not match the model features"}), 400
        # Preprocess the data
        preprocessor = Preprocessor(model_features=model_features, model_type=model, scaler_path=f"{model_folder_path}/scaler.pkl")
        features = preprocessor.create_encoded_features(features=extracted_features)

        pred = None
        conf = None

        if model == "ffnn":
            provider = FFNetProvider(f"{model_folder_path}/model.pt", len(model_features), config[model]["model_config"])
            # Load the model
            provider.load()

            # Predict
            pred, conf = provider.predict(features)
        if model == "rf":
            provider = RFProvider(f"{model_folder_path}/model.pkl", config[model]["model_config"])
            # Load the model
            provider.load()

            # Predict
            pred, conf = provider.predict(features)
        
        # Load training_data.json
        with open(f"exp/training_data.json", "r") as file:
            training_data = json.load(file)
        
        result = {"url": url, "features": extracted_features, "model": model, "pred": pred, "conf": conf, "trainingData": training_data}
        if debug == "true":
            result = {"url": url, "model": model, "html": html, "features": extracted_features, "pred": pred, "conf": conf, "trainingData": training_data}
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/multiple", methods=["POST"])
def predict_multiple():
    try:
        # Get the url and modesl from the request body
        url = request.json.get("url")
        models = request.json.get("models")
        debug = request.args.get("debug")
        voting_method = request.json.get("votingMethod")

        # Check if the url and models are valid
        if url is None:
            return jsonify({"error": "URL not provided"}), 400
        if models is None or models == []:
            # Use default models
            models = ["ffnn"]
        if not isinstance(models, list):
            return jsonify({"error": "Models must be a list"}), 400
        for model in models:
            if model not in ["ffnn", "rf"]:
                return jsonify({"error": "Model not found"}), 400

        # Crawl the html code    
        cralwer = WebCrawler()
        html = cralwer.crawl(url)
        if html.startswith("Error"):
            return jsonify({"error": html}), 400
        extractor = Extractor()
        extracted_features = extractor.extract_features(url, html)

        # Load config
        config_loader = ConfigLoader("config/config.json")
        config_loader.load()
        config = config_loader.get_config()

        model_results = []
        for model in models:
            model_features = config[model]["model_features"]
            model_version = config[model]["model_version"]
            model_folder_path = f'exp/models/{model}/{model_version}'

            # Check if extracted features match the model features
            if len(extracted_features) != len(model_features):
                return jsonify({"error": "Extracted features do not match the model features"}), 400
            # Preprocess the data
            preprocessor = Preprocessor(model_features=model_features, model_type=model, scaler_path=f"{model_folder_path}/scaler.pkl")
            features = preprocessor.create_encoded_features(features=extracted_features)

            pred = None
            conf = None
            provider = None
            if model == "ffnn":
                provider = FFNetProvider(f"{model_folder_path}/model.pt", len(model_features), config[model]["model_config"])
            if model == "rf":
                provider = RFProvider(f"{model_folder_path}/model.pkl", config[model]["model_config"])
            # Load the model
            provider.load()
            # Predict
            pred, conf = provider.predict(features)
            
            model_results.append({"model": model, "pred": pred, "conf": conf})
        
        # Load training_data.json
        with open(f"exp/training_data.json", "r") as file:
            training_data = json.load(file)
        
        # Do a vote
        pred = None
        if voting_method == "majority":
            # Get the most common prediction
            majority_vote_pred = mode([item['pred'] for item in model_results])
            pred = majority_vote_pred
        elif voting_method == "max-confidence":
            # Get the prediction with the highest confidence
            highest_conf_pred = max(model_results, key=lambda x:x['conf'])
            pred = highest_conf_pred['pred']
        elif voting_method == "average":
            # Get the average of all predictions
            average_pred = sum(item['pred'] for item in model_results) / len(model_results)
            pred = round(average_pred, 2)
        else:
            # Use the first model
            pred = model_results[0]['pred']

        result = {"url": url, "features": extracted_features, "pred": pred, "results": model_results, "trainingData": training_data}
        if debug == "true":
            result = {"url": url, "pred": pred, "results": model_results, "html": html, "features": extracted_features, "trainingData": training_data}
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)