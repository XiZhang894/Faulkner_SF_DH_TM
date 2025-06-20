# extreme_detection_and_snippet.py
# Detect extreme and sudden-change windows, then extract corresponding text snippets

import os
import pandas as pd
import spacy

# === Settings === #
STYLE_CSV = "style_metrics_sliding_window_full.csv"
CORPUS_DIR = "corpus"
WINDOW_SIZE = 200  # must match the feature extraction window

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# 1. Load sliding-window feature data
df = pd.read_csv(STYLE_CSV)
metrics = ["MSL", "SCR", "TTR", "AWL"]  # example metrics
k = 2  # threshold multiplier for std-dev detection

# Prepare storage for detected positions
extreme_positions = {m: [] for m in metrics}
change_positions = {m: [] for m in metrics}

# 2. Detect extremes and sudden changes
for m in metrics:
    series = df[m]
    mu, sigma = series.mean(), series.std()

    # extremes: beyond mean ± k*std
    mask_extreme = (series > mu + k*sigma) | (series < mu - k*sigma)
    extreme_positions[m] = df.loc[mask_extreme, "window_start"].tolist()

    # changes: abs diff > mean(diff) + k*std(diff)
    diffs = series.diff().abs()
    mu_d, sigma_d = diffs.mean(), diffs.std()
    mask_change = diffs > (mu_d + k*sigma_d)
    change_positions[m] = df.loc[mask_change.fillna(False), "window_start"].tolist()

# 3. Define snippet extraction function
def extract_window_text(section_file, window_start, window_size=WINDOW_SIZE):
    """
    Given a section XML file and a word-index start, return the text snippet of window_size words.
    """
    # read paragraphs
    lines = []
    with open(section_file, encoding='utf-8') as f:
        for line in f:
            if '<p>' in line:
                lines.append(line.replace('<p>', '').replace('</p>', '').strip())
    full_text = ' '.join(lines)

    # tokenize
    doc = nlp(full_text)
    words = [tok.text for tok in doc if tok.is_alpha]

    # slice window
    snippet_words = words[window_start: window_start + window_size]
    return ' '.join(snippet_words)

# 4. Extract and save snippets for each detected point
output = []
for m in metrics:
    for pos in sorted(set(extreme_positions[m] + change_positions[m])):
        # determine section by matching df row
        row = df[df['window_start'] == pos].iloc[0]
        section = row['section']
        xml_path = os.path.join(CORPUS_DIR, f"{section}.xml")
        snippet = extract_window_text(xml_path, pos)
        output.append({
            'metric': m,
            'window_start': pos,
            'section': section,
            'snippet': snippet
        })

# Save results as CSV
out_df = pd.DataFrame(output)
out_df.to_csv("detected_snippets.csv", index=False)
print("Saved detected_snippets.csv with extracted text snippets")
