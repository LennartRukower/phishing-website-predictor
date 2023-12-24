from flask import Flask
from flask import request
from services.WebCrawler import WebCrawler
from models.FFNetProvider import FFNetProvider
from services.Preprocessor import Preprocessor
from services.Extractor import Extractor
from config.ConfigLoader import ConfigLoader

app = Flask(__name__)

@app.route("/models")
def get_models():
    return ["fnn"]

@app.route("/predict", methods=["POST"])
def predict_url():
    # Get the url and model from the request
    url = request.form.get("url")
    model = request.form.get("model")

    # Check if the url and model are valid
    if url is None:
        return {"error": "URL not provided"}
    if model is None:
        # Use default model
        model = "fnn"
    if model not in ["fnn"]:
        return {"error": "Model not found"}

    # Crawl the html code    
    cralwer = WebCrawler()
    html = cralwer.crawl(url)

    # Load config
    config_loader = ConfigLoader("./config.json")
    config = config_loader.get_config()
    model_features = config["model_features"]

    extractor = Extractor()
    extracted_features = extractor.extract_features(url, html)

    # Check if extracted features match the model features
    if len(extracted_features) != len(model_features):
        return {"error": "Extracted features do not match the model features"}
    # Preprocess the data
    preprocessor = Preprocessor(model_features=model_features, model_type='FFNet', scaler_path='./exp/scaler.pkl')
    features = preprocessor.create_encoded_features(features=extracted_features)

    # Predict
    if (model == "fnn"):
        provider = FFNetProvider("exp/models/2023-12-10-model.pt", len(model_features))
        # Load the model
        provider.load()
        
        # Predict
        pred = provider.predict(features)
        pass

    result = {"url": url, "model": model, "html": html, "pred": pred}
    return result

if __name__ == "__main__":
    app.run(debug=True)