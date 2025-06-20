# feature_extraction.py
# Expanded stylistic feature extraction including advanced metrics

import os
import spacy
import pandas as pd
from lexicalrichness import LexicalRichness
from collections import Counter
import textstat  # pip install textstat

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Define subordinate clause counter using dependency parsing
def count_subordinate_clauses(doc):
    return sum(1 for tok in doc if tok.dep_ == "mark" and tok.head.pos_ == "VERB")


# Define passive voice ratio
def passive_ratio(doc):
    subclauses = list(doc.sents)
    if not subclauses:
        return 0
    auxpass_count = sum(1 for tok in doc if tok.dep_ == "auxpass")
    return auxpass_count / len(subclauses)


# Define mean clause length (MCL)
def mean_clause_length(doc):
    sentences = list(doc.sents)
    clauses = []
    for sent in sentences:
        parts = sent.text.replace(';', ',').split(',')
        clauses.extend([p.strip() for p in parts if p.strip()])
    if not clauses:
        return 0
    return sum(len(clause.split()) for clause in clauses) / len(clauses)


# Process each section file
sections = [f.replace('.xml', '') for f in os.listdir('corpus') if f.endswith('.xml')]
results = []

for section in sections:
    path = os.path.join('corpus', f'{section}.xml')
    text = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '<p>' in line:
                # strip XML tags
                content = line.replace('<p>', '').replace('</p>', '').strip()
                text.append(content)
    full_text = ' '.join(text)
    doc = nlp(full_text)
    tokens = [tok for tok in doc if tok.is_alpha]
    num_sents = len(list(doc.sents))
    num_tokens = len(tokens)

    # Basic metrics
    msl = num_tokens / num_sents if num_sents else 0
    scr = count_subordinate_clauses(doc) / num_sents if num_sents else 0
    ttr = len(set(tok.lemma_.lower() for tok in tokens)) / num_tokens if num_tokens else 0
    awl = sum(len(tok.text) for tok in tokens) / num_tokens if num_tokens else 0

    # Advanced metrics
    pr = passive_ratio(doc)
    mcl = mean_clause_length(doc)
    lex = LexicalRichness(full_text)
    mtld = lex.mtld()
    flesch = textstat.flesch_reading_ease(full_text)

    # POS distribution
    pos_counts = Counter(tok.pos_ for tok in tokens)
    pos_dist = {f"POS_{pos}": count / num_tokens for pos, count in pos_counts.items()}

    # Compile result
    result = {
        "section": section,
        "MSL": round(msl, 3),
        "SCR": round(scr, 3),
        "TTR": round(ttr, 3),
        "AWL": round(awl, 3),
        "PassiveRatio": round(pr, 3),
        "MCL": round(mcl, 3),
        "MTLD": round(mtld, 3),
        "Flesch": round(flesch, 3)
    }
    result.update(pos_dist)
    results.append(result)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('style_metrics.csv', index=False)
print('Saved expanded style_metrics.csv')
print(df)
