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
    return ["ffnn"]

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
        model = "ffnn"
    if model not in ["ffnn"]:
        return {"error": "Model not found"}

    # Crawl the html code    
    cralwer = WebCrawler()
    html = cralwer.crawl(url)
    if (html.startswith("Error")):
        return {"error": html}

    # Load config
    config_loader = ConfigLoader("./config/config.json")
    config_loader.load()
    config = config_loader.get_config()
    model_features = config[model]["model_features"]
    model_folder_path = "exp/models/2023-12-24_3"

    extractor = Extractor()
    extracted_features = extractor.extract_features(url, html)

    # Check if extracted features match the model features
    if len(extracted_features) != len(model_features):
        return {"error": "Extracted features do not match the model features"}
    # Preprocess the data
    preprocessor = Preprocessor(model_features=model_features, model_type='FFNet', scaler_path=f"{model_folder_path}/scaler.pkl")
    features = preprocessor.create_encoded_features(features=extracted_features)

    # Predict
    if (model == "ffnn"):
        provider = FFNetProvider(f"{model_folder_path}/model.pt", len(model_features))
        # Load the model
        provider.load()
        
        # Predict
        pred = provider.predict(features)
        pass

    result = {"url": url, "model": model, "html": html, "ex_fet": extracted_features, "pred": pred}
    return result

if __name__ == "__main__":
    app.run(debug=True)