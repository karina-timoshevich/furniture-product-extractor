import os
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForTokenClassification

MODEL_NAME = "bert-base-cased"
DATA_FILE = "ner_dataset_final.txt"
OUTPUT_DIR = "./ner_model"

def load_data(filepath):
    sentences = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
            else:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)
    return sentences, labels


sentences, tags = load_data(DATA_FILE)

unique_labels = sorted(set(tag for doc in tags for tag in doc))
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}

tags_ids = [[label2id[tag] for tag in doc] for doc in tags]
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForTokenClassification(tokenizer)


def tokenize_and_align(examples):
    tokenized_inputs = tokenizer( examples["tokens"], truncation=True, is_split_into_words=True,)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                prev_label = label[word_idx]
                if id2label[prev_label].startswith("B-"):
                    label_ids.append(label2id["I-PRODUCT"])
                else:
                    label_ids.append(prev_label)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = Dataset.from_dict({"tokens": sentences, "labels": tags_ids})
dataset = dataset.map(tokenize_and_align, batched=True)

model = BertForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_steps=500,
    logging_strategy="steps",
    logging_steps=20,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
