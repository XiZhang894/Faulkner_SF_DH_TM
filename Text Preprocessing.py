# text_preprocessing.py
# This script preprocesses the four TEI-annotated narrative sections of Faulkner's "The Sound and the Fury"
# by performing tokenization, lemmatization, POS tagging, and dependency parsing using spaCy.
# It filters stopwords and low-frequency lemmas for downstream tasks.

import spacy
import json
from collections import Counter
from nltk.corpus import stopwords
import os
import nltk

nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Custom stopwords (Faulkner-specific additions)
custom_stopwords = {"'em", "'bout"}
stop_words = set(stopwords.words('english')).union(custom_stopwords)

# Load and process each section
input_dir = "corpus"
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

lemma_counter = Counter()


# First pass: collect lemma frequency from all documents
def collect_lemmas():
    for file in os.listdir(input_dir):
        if file.endswith(".xml"):
            with open(os.path.join(input_dir, file), encoding="utf-8") as f:
                text = f.read()
                text_content = ' '.join([line.strip() for line in text.splitlines() if '<p>' in line])
                doc = nlp(text_content)
                lemma_counter.update([token.lemma_ for token in doc if token.is_alpha])


# Second pass: preprocess and export
def preprocess_documents():
    for file in os.listdir(input_dir):
        if file.endswith(".xml"):
            section = file.replace(".xml", "")
            with open(os.path.join(input_dir, file), encoding="utf-8") as f:
                text = f.read()
                text_content = ' '.join([line.strip() for line in text.splitlines() if '<p>' in line])
                doc = nlp(text_content)
                output = []
                for i, sent in enumerate(doc.sents):
                    tokens, lemmas, pos = [], [], []
                    for token in sent:
                        if token.is_alpha and token.lemma_ not in stop_words and lemma_counter[token.lemma_] >= 3:
                            tokens.append(token.text)
                            lemmas.append(token.lemma_)
                            pos.append(token.pos_)
                    if tokens:
                        output.append({
                            "section": section,
                            "sentence_id": i,
                            "sentence": sent.text,
                            "tokens": tokens,
                            "lemmas": lemmas,
                            "pos": pos
                        })
                out_path = os.path.join(output_dir, f"{section}.jsonl")
                with open(out_path, "w", encoding="utf-8") as out:
                    for entry in output:
                        out.write(json.dumps(entry) + "\n")
                print(f"Processed: {section} → {out_path}")


if __name__ == "__main__":
    collect_lemmas()
    preprocess_documents()
