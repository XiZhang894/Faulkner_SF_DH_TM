import os
import spacy
import pandas as pd
from lexical_diversity import lex_div as ld

# Load the spaCy model (make sure it includes parser and sentencizer)
nlp = spacy.load("en_core_web_sm")

# Add sentencizer if not already present (usually en_core_web_sm has a parser, so this may be optional)
if 'sentencizer' not in nlp.pipe_names:
    nlp.add_pipe('sentencizer')

window_size = 500
step_size = 100


def calc_subordinate_clause_ratio(doc):
    total_clauses = 0
    subordinate_clauses = 0
    for token in doc:
        # A marker 'mark' with a verbal head indicates a subordinate clause
        if token.dep_ == "mark" and token.head.pos_ == "VERB":
            subordinate_clauses += 1
        if token.pos_ == "VERB":
            total_clauses += 1
    return subordinate_clauses / total_clauses if total_clauses > 0 else 0


def calc_passive_aux_ratio(doc):
    auxpass = sum(1 for token in doc if token.dep_ == "auxpass")
    verbs = sum(1 for token in doc if token.pos_ == "VERB")
    return auxpass / verbs if verbs > 0 else 0


def calc_past_participle_ratio(doc):
    vbn = sum(1 for token in doc if token.tag_ == "VBN")
    verbs = sum(1 for token in doc if token.pos_ == "VERB")
    return vbn / verbs if verbs > 0 else 0


def calc_type_token_ratio(tokens):
    types = set(tokens)
    return len(types) / len(tokens) if tokens else 0


def calc_avg_word_length(tokens):
    lengths = [len(t) for t in tokens]
    return sum(lengths) / len(lengths) if lengths else 0


def calc_pos_ratios(doc):
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    total = 0
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
        total += 1
    if total == 0:
        return {k: 0 for k in pos_counts}
    return {k: v / total for k, v in pos_counts.items()}


def calc_mtld(tokens):
    try:
        return ld.mtld(tokens)
    except Exception:
        return 0


def calc_avg_clause_length(doc):
    clause_lengths = []
    for sent in doc.sents:
        clause_lengths.append(len(sent))
    return sum(clause_lengths) / len(clause_lengths) if clause_lengths else 0


def calc_mean_dependency_distance(doc):
    distances = []
    for token in doc:
        if token.head.i != token.i:
            distances.append(abs(token.i - token.head.i))
    return sum(distances) / len(distances) if distances else 0


def calc_subordination_index(doc):
    sents = list(doc.sents)
    subordinated_sents = 0
    for sent in sents:
        # Check if sentence contains subordinating markers
        if any(token.dep_ == "mark" for token in sent):
            subordinated_sents += 1
    return subordinated_sents / len(sents) if sents else 0


results = []

for filename in os.listdir("corpus"):
    if filename.endswith(".xml"):
        section = filename.replace(".xml", "")
        with open(os.path.join("corpus", filename), encoding="utf-8") as f:
            text_lines = []
            for line in f:
                if '<p>' in line:
                    content = line.replace('<p>', '').replace('</p>', '').strip()
                    text_lines.append(content)
        full_text = " ".join(text_lines)
        doc = nlp(full_text)
        tokens = [token.text for token in doc if token.is_alpha]

        for start in range(0, len(tokens) - window_size + 1, step_size):
            window_tokens = tokens[start: start + window_size]
            window_doc = spacy.tokens.Doc(nlp.vocab, words=window_tokens)

            # Manually run sentencizer and parser on the created Doc to get sentence boundaries and dependencies
            if 'sentencizer' in nlp.pipe_names:
                nlp.get_pipe('sentencizer')(window_doc)
            if 'parser' in nlp.pipe_names:
                nlp.get_pipe('parser')(window_doc)

            # Calculate metrics
            sent_lens = [len(sent) for sent in window_doc.sents]
            msl = sum(sent_lens) / len(sent_lens) if sent_lens else 0
            scr = calc_subordinate_clause_ratio(window_doc)
            passive_aux = calc_passive_aux_ratio(window_doc)
            past_participle = calc_past_participle_ratio(window_doc)
            ttr = calc_type_token_ratio(window_tokens)
            awl = calc_avg_word_length(window_tokens)
            pos_ratios = calc_pos_ratios(window_doc)
            mtld = calc_mtld(window_tokens)
            avg_clause_len = calc_avg_clause_length(window_doc)
            mean_dep_dist = calc_mean_dependency_distance(window_doc)
            subord_index = calc_subordination_index(window_doc)

            results.append({
                "section": section,
                "window_start": start,
                "MSL": msl,
                "SCR": scr,
                "PassiveAuxRatio": passive_aux,
                "PastParticipleRatio": past_participle,
                "TTR": ttr,
                "AWL": awl,
                "MTLD": mtld,
                "NounRatio": pos_ratios["NOUN"],
                "VerbRatio": pos_ratios["VERB"],
                "AdjRatio": pos_ratios["ADJ"],
                "AdvRatio": pos_ratios["ADV"],
                "AvgClauseLength": avg_clause_len,
                "MeanDependencyDistance": mean_dep_dist,
                "SubordinationIndex": subord_index
            })

df = pd.DataFrame(results)
df.to_csv("style_metrics_sliding_window_full.csv", index=False)
print("Saved style_metrics_sliding_window_full.csv")
