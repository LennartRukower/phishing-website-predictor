from flask import Flask

app = Flask(__name__)

@app.route("/models")
def get_models():
    return ["nn", "svm", "rf"]

@app.route("/predict")
def predict_url():
    result = {"is_phishing": True}
    return result

if __name__ == "__main__":
    app.run(debug=True)