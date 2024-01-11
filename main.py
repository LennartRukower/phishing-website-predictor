from flask import Flask
from flask import request
from services.WebCrawler import WebCrawler

app = Flask(__name__)

@app.route("/models")
def get_models():
    return ["fnn"]



if __name__ == "__main__":
    app.run(debug=True)