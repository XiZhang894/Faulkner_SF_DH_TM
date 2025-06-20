# text_mining_analysis.py
# Performs Sentiment Analysis and Topic Modeling on Faulkner's narrative sections

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from gensim import corpora, models
from nltk import download
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.spatial.distance import jensenshannon

# Ensure VADER lexicon is available
download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


# === Load Tokens from JSONL === #
def get_tokens_from_jsonl(path):
    token_list = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            token_list.extend(entry['tokens'])
    return token_list


# === Sentiment: Sliding Window === #
def windowed_sentiment(token_list, window_size=500, overlap=100):
    scores = []
    for win_idx in range(0, len(token_list) - window_size + 1, window_size - overlap):
        window = token_list[win_idx:win_idx + window_size]
        text = ' '.join(window)
        score = sid.polarity_scores(text)
        scores.append(score['compound'])
    return scores


# === Topic Modeling: Token Windows === #
def build_token_windows(token_list, window_size=500, overlap=100):
    win_list = []
    for win_idx in range(0, len(token_list) - window_size + 1, window_size - overlap):
        win = token_list[win_idx:win_idx + window_size]
        win_list.append(win)
    return win_list


# === Main Processing === #
sentiment_results = {}
topic_windows = []
section_labels = []
section_window_map = defaultdict(list)

for file in os.listdir('processed'):
    if file.endswith('.jsonl'):
        section = file.replace('.jsonl', '')
        token_list = get_tokens_from_jsonl(os.path.join('processed', file))

        # Sentiment arcs
        arc = windowed_sentiment(token_list)
        sentiment_results[section] = arc

        # Topic modeling windows
        win_list = build_token_windows(token_list)
        topic_windows.extend(win_list)
        section_labels.extend([section] * len(win_list))
        section_window_map[section].extend(win_list)

# === Save Sentiment Arc Plot === #
plt.figure(figsize=(10, 6))
for section, arc in sentiment_results.items():
    plt.plot(arc, label=section)
plt.title("Emotional Arcs by Narrative Section")
plt.xlabel("Window Index")
plt.ylabel("Sentiment Score (-1 to +1)")
plt.legend()
plt.tight_layout()
plt.savefig("sentiment_arcs.png")
print("Saved: sentiment_arcs.png")

# === Train LDA Model === #
dictionary = corpora.Dictionary(topic_windows)
corpus = [dictionary.doc2bow(win) for win in topic_windows]
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10, random_state=42)

# === Extract Topic Distributions === #
topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
topic_matrix = np.zeros((len(topic_distributions), 5))
for row_index, topic_dist in enumerate(topic_distributions):
    for topic_id, prob in topic_dist:
        topic_matrix[row_index][topic_id] = prob

df_topic = pd.DataFrame(topic_matrix, columns=[f'Topic_{i}' for i in range(5)])
df_topic['section'] = section_labels
df_topic.to_csv('topic_windows.csv', index=False)
print("Saved: topic_windows.csv")

# === Heatmap by section (mean topic proportions) === #
heatmap_df = df_topic.groupby('section').mean().round(3)
plt.figure(figsize=(8, 4))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu")
plt.title("Average Topic Distribution per Section")
plt.tight_layout()
plt.savefig("topic_heatmap.png")
print("Saved: topic_heatmap.png")

# === Jensen-Shannon divergence (topic shift rate) === #
shift_scores = []
for i in range(1, len(topic_matrix)):
    js = jensenshannon(topic_matrix[i - 1], topic_matrix[i])
    shift_scores.append(js)

plt.figure(figsize=(10, 4))
plt.plot(shift_scores)
plt.title("Topic Shift Rate (Jensen-Shannon Divergence)")
plt.xlabel("Window Index")
plt.ylabel("JS Divergence")
plt.tight_layout()
plt.savefig("topic_shift_curve.png")
print("Saved: topic_shift_curve.png")

# === Top keywords per topic === #
with open("topic_keywords.txt", "w", encoding="utf-8") as f:
    for i in range(5):
        f.write(f"Topic {i}:\n")
        keywords = lda_model.show_topic(i, topn=10)
        for word, weight in keywords:
            f.write(f"  {word:10s} {weight:.4f}\n")
        f.write("\n")
print("Saved: topic_keywords.txt")

# === Poster/Report Export (PDF Summary) === #
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Times", 'B', 14)
pdf.cell(0, 10, "Faulkner Text-Mining Summary Report", ln=True, align='C')

pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 10,
               "This report summarizes the sentiment and topic modeling analysis for each narrative section in "
               "William Faulkner's 'The Sound and the Fury'. It includes emotional arcs, average topic distributions, "
               "and topic transitions.")

pdf.ln(5)
pdf.image("sentiment_arcs.png", w=180)
pdf.ln(5)
pdf.image("topic_heatmap.png", w=180)
pdf.ln(5)
pdf.image("topic_shift_curve.png", w=180)

pdf.add_page()
pdf.set_font("Times", 'B', 12)
pdf.cell(0, 10, "Top 10 Keywords per Topic:", ln=True)
pdf.set_font("Times", '', 10)
with open("topic_keywords.txt", encoding="utf-8") as f:
    for line in f:
        pdf.multi_cell(0, 5, line.strip())

pdf.output("faulkner_analysis_summary.pdf")
print("Saved: faulkner_analysis_summary.pdf")
