from flask import Flask, render_template, request
from scraper import extract_text_from_url
from ner_predict import ner_predictor

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
