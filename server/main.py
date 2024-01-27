from flask import Flask, jsonify
from flask import request
from services.WebCrawler import WebCrawler
from models.FFNetProvider import FFNetProvider
from services.Preprocessor import Preprocessor
from services.Extractor import Extractor
from config.ConfigLoader import ConfigLoader
from models.RFProvider import RFProvider
import json

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

        result = {"url": url, "features": extracted_features, "model": model, "pred": pred, "conf": conf}
        if debug == "true":
            result = {"url": url, "model": model, "html": html, "features": extracted_features, "pred": pred, "conf": conf}
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)