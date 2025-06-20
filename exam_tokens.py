import spacy

nlp = spacy.load("en_core_web_sm")

with open("corpus/April eighth, 1928.xml", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)
tokens = [token.text for token in doc if token.is_alpha]

print(f"Total tokens in section: {len(tokens)}")

window_size = 200
step_size = 100
num_windows = (len(tokens) - window_size) // step_size + 1
print(f"Number of sliding windows: {num_windows}")

# Print first 3 windows tokens (only first 20 tokens each)
for i in range(min(3, num_windows)):
    window_tokens = tokens[i*step_size : i*step_size + window_size]
    print(f"Window {i+1} tokens sample: {window_tokens[:20]}")
