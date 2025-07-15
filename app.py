from flask import Flask, render_template, request
from scraper import extract_text_from_url
from ner_predict import ner_predictor
import os
import zipfile
import gdown


def download_model():
    if not os.path.exists("ner_model_1"):
        print("Downloading model from Google Drive...")
        file_id = "1sOrOWSPSVWUAvyQICO6igY9ts1kAFSBw"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "ner_model_1.zip", quiet=False)
        with zipfile.ZipFile("ner_model_1.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Model unpacked!")

download_model()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    products = None
    url = ""
    error = None

    if request.method == "POST":
        url = request.form["url"]
        text = extract_text_from_url(url)
        if text:
            products = ner_predictor.extract_products(text)
        else:
            error = "Не удалось получить текст с указанного URL."

    return render_template(
        "index.html",
        products=products,
        url=url,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
