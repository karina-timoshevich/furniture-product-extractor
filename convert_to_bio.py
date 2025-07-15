import re


def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)


def convert_tagged_to_bio(text):
    bio_data = []
    while "<PRODUCT>" in text:
        pre, tagged_part = text.split("<PRODUCT>", 1)
        tagged_text, post = tagged_part.split("</PRODUCT>", 1)
        tokens_pre = tokenize(pre)
        tokens_tagged = tokenize(tagged_text)
        bio_data.extend([(tok, "O") for tok in tokens_pre])

        for i, tok in enumerate(tokens_tagged):
            tag = "B-PRODUCT" if i == 0 else "I-PRODUCT"
            bio_data.append((tok, tag))
        text = post

    bio_data.extend([(tok, "O") for tok in tokenize(text)])
    return bio_data


if __name__ == "__main__":
    with open("data/annotated_raw.txt", "r", encoding="utf-8") as f:
        raw = f.read()
    bio = convert_tagged_to_bio(raw)

    with open("ner_dataset.txt", "w", encoding="utf-8") as out:
        for token, label in bio:
            out.write(f"{token} {label}\n")
        out.write("\n")
