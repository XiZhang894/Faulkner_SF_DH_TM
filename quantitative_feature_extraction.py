# feature_extraction.py
# Compute MSL, SCR, TTR, and AWL from preprocessed Faulkner section data

import json
import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt

# Set Font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load data from JSONL files
data = []
for fname in os.listdir('processed'):
    if fname.endswith('.jsonl'):
        with open(os.path.join('processed', fname), encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

df = pd.DataFrame(data)

# === MSL: Mean Sentence Length === #
df['sentence_length'] = df['tokens'].apply(len)
msl = df.groupby('section')['sentence_length'].mean().rename('MSL')


# === SCR: Subordinate Clause Ratio === #
def count_sub_clauses(text):
    doc = nlp(text)
    return sum(1 for tok in doc if tok.dep_ == 'mark')


df['subordinate_clauses'] = df['sentence'].apply(count_sub_clauses)
scr = df.groupby('section')['subordinate_clauses'].sum() / df.groupby('section')['sentence_id'].count()
scr.name = 'SCR'


# === TTR: Type–Token Ratio === #
def compute_ttr(token_lists):
    tokens = [tok for sublist in token_lists for tok in sublist]
    return len(set(tokens)) / len(tokens) if tokens else 0


ttr = df.groupby('section')['tokens'].apply(compute_ttr).rename('TTR')


# === AWL: Average Word Length === #
def avg_word_length(token_lists):
    tokens = [tok for sublist in token_lists for tok in sublist if tok.isalpha()]
    return sum(len(tok) for tok in tokens) / len(tokens) if tokens else 0


awl = df.groupby('section')['tokens'].apply(avg_word_length).rename('AWL')

# === Combine & Save === #
features = pd.concat([msl, scr, ttr, awl], axis=1).round(3)
features.to_csv('features_summary.csv')

# === Visualize === #
features.plot.bar(rot=0, figsize=(8, 5), title='Narrative Style Metrics')
plt.ylabel("Value")

# Add value labels: display values on top of each bar
ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge')  # Automatically show each bar's height at the top

plt.tight_layout()
plt.savefig("feature_metrics.png")
print("Saved features_summary.csv and feature_metrics.png")

