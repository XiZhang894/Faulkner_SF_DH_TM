import json
import os

import spacy

# spaCy English模型
nlp = spacy.load("en_core_web_sm")

# 预定义简易情感词典（这里示例用简单词表，实际可替换为更丰富词典）
positive_words = {"good", "happy", "love", "excellent", "fortunate", "correct", "superior"}
negative_words = {"bad", "sad", "hate", "terrible", "unfortunate", "wrong", "inferior"}

# 滑动窗口参数
window_size = 500
step_size = 100


def compute_sentiment_score(doc):
    # doc 是 spaCy 处理后的 Doc 对象
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    score = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, score))  # 限制在[-1,1]


sentiment_arcs = {}

for filename in os.listdir("corpus"):
    if filename.endswith(".xml"):
        section = filename.replace(".xml", "")
        with open(os.path.join("corpus", filename), encoding="utf-8") as f:
            text = []
            for line in f:
                if '<p>' in line:
                    content = line.replace('<p>', '').replace('</p>', '').strip()
                    text.append(content)
        full_text = " ".join(text)
        doc = nlp(full_text)

        # 获取所有词汇（仅alpha词）
        tokens = [token for token in doc if token.is_alpha]

        arc = []
        for start_idx in range(0, len(tokens) - window_size + 1, step_size):
            window_tokens = tokens[start_idx: start_idx + window_size]
            window_doc = spacy.tokens.Doc(doc.vocab, words=[t.text for t in window_tokens])
            score = compute_sentiment_score(window_doc)
            arc.append(round(score, 4))

        sentiment_arcs[section] = arc

with open("sentiment_arcs.json", "w", encoding="utf-8") as f:
    json.dump(sentiment_arcs, f, indent=2)

print("Saved sentiment_arcs.json")
