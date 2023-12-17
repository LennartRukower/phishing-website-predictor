from flask import Flask
from flask import request
from services.WebCrawler import WebCrawler
from models.FFNetProvider import FFNetProvider

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
    cralwer = WebCrawler(url)
    html = cralwer.crawl()

    # Extract the features
    model_features = [
        "SubdomainLevel",
        "UrlLength",
        "NumDashInHostname",
        "TildeSymbol",
        "NumPercent",
        "NumAmpersand",
        "NumNumericChars",
        "DomainInSubdomains",
        "HttpsInHostname",
        "PathLength",
        "DoubleSlashInPath",
        "PctExtResourceUrls",
        "InsecureForms",
        "ExtFormAction",
        "PopUpWindow",
        "IframeOrFrame",
        "ImagesOnlyInForm",
    ]

    # Preprocess the data
    features = None

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