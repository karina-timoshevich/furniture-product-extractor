import os
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline

MODEL_PATH = "./ner_model_1"
TEST_TEXT = """Submit Close search Home Catalog Hari Raya Furniture Sale Extra Discount Search Latitude Pay Scan QR Code ( Last 3 Day ) Open Giant SALE / / Open July 1 to 6 , 2025 / 10 am to 7 pm // Tax Free / Up to 70% Off Clearance / Extra 10% OFF,  Free Instalment, 0% Interest! Home Catalog Hari Raya Furniture Sale Extra Discount Search Latitude Pay Scan QR Code Submit Search Cart Cart expand/collapse ASMUND Outdoor Console Table with 2 Sink Set Regular price$4,199.00$1,399.00Sale Regular price $1,399.00Sale $1,399.00 Sale Add to cart Asmund Outdoor Console Table with 2 Sink Set 1TF168INX CONSOLE Table 2 SINK Set Share Share on Facebook Tweet Tweet on Twitter Pin it Pin on Pinterest Home Catalog Hari Raya Furniture Sale Extra Discount Search Latitude Pay Scan QR Code Hari Raya Ramadan Teak Wood Furniture Sale Search Buy Now Pay Later Subscribe Subscribe
"""

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
entities = ner_pipeline(TEST_TEXT)


def is_valid_product(ent):
    text = ent['word'].strip()
    score = ent['score']
    if score < 0.90:
        return False
    if len(text) < 5:
        return False
    if len(text.split()) < 2:
        return False
    if any(tok in text.lower() for tok in ['search', 'account', 'blog', 'cart', 'sign', 'create']):
        return False
    return True

filtered_entities = [ent for ent in entities if is_valid_product(ent)]

print("=== Обнаруженные сущности ===")
for ent in filtered_entities:
    print(f"{ent['word']} ({ent['entity_group']}) — score: {ent['score']:.2f}")

