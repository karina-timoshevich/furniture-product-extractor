from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
from difflib import SequenceMatcher

MODEL_PATH = "./ner_model_1"

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

def clean_token(text):
    return text.replace("##", "").strip()


def is_valid_product(ent):
    text = ent['word'].strip()
    score = ent['score']

    if score < 0.90:
        return False
    if len(text) < 5:
        return False
    if len(text.split()) < 2:
        return False

    blacklist = [
        'wishlist', 'help center', 'menu', 'call us', 'product categories',
        'search', 'account', 'blog', 'cart', 'sign', 'create',
        'submit', 'newsletter', 'login', 'facebook', 'twitter', 'sale',
        'subscribe', 'instalment', 'interest', 'discount',
        'software', 'blocker', 'shipping', 'policy', 'free', 'contact'
    ]
    if any(tok in text.lower() for tok in blacklist):
        return False
    if "£" in text or "$" in text or "€" in text:
        return False
    if any(text.lower().endswith(x) for x in ["upholstery", "fabric"]) and "bed" not in text.lower():
        return False
    return True


def remove_similar(products):
    result = []
    for p in sorted(products, key=len, reverse=True):
        if not any(SequenceMatcher(None, p, r).ratio() > 0.85 for r in result):
            result.append(p)
    return result


def remove_duplicate_phrases(text, min_len=5):
    words = text.split()
    length = len(words)
    seen_phrases = set()
    result = []
    i = 0
    while i < length:
        matched = False
        for size in range(min(12, length - i), min_len - 1, -1):
            phrase = " ".join(words[i:i + size])
            if phrase in seen_phrases:
                i += size
                matched = True
                break
            seen_phrases.add(phrase)
        if not matched:
            result.append(words[i])
            i += 1
    return " ".join(result)

def remove_similar(products):
    result = []
    for p in sorted(products, key=len, reverse=True):
        if not any(SequenceMatcher(None, p, r).ratio() > 0.85 for r in result):
            result.append(p)
    return result


class NERPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        self.model = BertForTokenClassification.from_pretrained(MODEL_PATH)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def extract_products(self, text):
        entities = self.ner_pipeline(text)
        clean = [
            clean_token(ent['word'])
            for ent in entities
            if ent['entity_group'] == 'PRODUCT' and is_valid_product(ent)
        ]
        cleaned = [remove_duplicate_phrases(p) for p in clean]
        return sorted(set(remove_similar(cleaned)))

ner_predictor = NERPredictor()
