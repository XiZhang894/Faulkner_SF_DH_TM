import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Step 1: register the TTF
font_path = "times.ttf"  # path to your .ttf
prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)

# Set global font
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.size'] = 12

# Raw text data
raw_data = """
Topic 0:
  said       0.0823
  p          0.0735
  I          0.0717
  Caddy      0.0221
  me         0.0181
  Dilsey     0.0133
  go         0.0124
  going      0.0108
  went       0.0106
  Mother     0.0088

Topic 1:
  said       0.0717
  I          0.0422
  Luster     0.0365
  Mother     0.0229
  Quentin    0.0170
  going      0.0147
  p          0.0128
  know       0.0127
  got        0.0121
  Dilsey     0.0119

Topic 2:
  said       0.0012
  I          0.0009
  p          0.0009
  Luster     0.0009
  Dilsey     0.0009
  went       0.0008
  Caddy      0.0008
  going      0.0008
  me         0.0008
  Quentin    0.0008

Topic 3:
  I          0.1625
  says       0.0982
  p          0.0333
  me         0.0160
  know       0.0139
  got        0.0131
  went       0.0116
  right      0.0114
  back       0.0090
  Dilsey     0.0090

Topic 4:
  said       0.0791
  Dilsey     0.0487
  I          0.0396
  Luster     0.0283
  p          0.0263
  Jason      0.0250
  hit        0.0178
  de         0.0178
  dat        0.0121
  Compson    0.0114
"""

# Parse raw data
records = []
current_topic = None
for line in raw_data.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("Topic"):
        current_topic = int(line.split()[1].strip(':'))
    else:
        parts = line.split()
        word = parts[0]
        prob = float(parts[1])
        records.append({"Topic": current_topic, "Word": word, "Probability": prob})

# Create DataFrame and pivot
df = pd.DataFrame(records)
wide_df = df.pivot(index='Word', columns='Topic', values='Probability')
wide_df = wide_df[[0, 1, 2, 3, 4]].round(4)
# Select top 10 words by Topic 0 as example
top_words = wide_df.sort_values(0, ascending=False).head(10)

# Generate table image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')  # Hide axes

# Create table with Times New Roman
table = ax.table(
    cellText=top_words.values,
    rowLabels=top_words.index,
    colLabels=[f"Topic {i}" for i in top_words.columns],
    cellLoc='center',
    rowLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Set title with Times New Roman
plt.title("Top Keywords and Frequencies by Topic", fontweight='bold', fontsize=14, family='Times New Roman', pad=20)

# Save as image
output_path = "topic_keywords_wide_table.png"
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Saved wide-format topic table to {output_path}")
