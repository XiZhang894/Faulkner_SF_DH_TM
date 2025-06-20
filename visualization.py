import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import uniform_filter1d
import json
import numpy as np
from fpdf import FPDF

# === 3.5.1 Stylistic Feature Visualization & ANOVA === #

df_style = pd.read_csv("features_summary.csv")  # 包含 section, MSL, SCR, TTR, AWL

sns.set(style="whitegrid")
metrics = ["MSL", "SCR", "TTR", "AWL"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="section", y=metric, data=df_style, palette="Set2")
    plt.title(f"{metric} by Narrative Section")
    plt.xlabel("Section")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{metric}_by_section.png")
    plt.close()
    print(f"Saved: {metric}_by_section.png")

print("\n--- ANOVA Results for Stylistic Metrics ---")
for metric in metrics:
    groups = [group[metric].values for _, group in df_style.groupby("section")]
    f_stat, p_val = f_oneway(*groups)
    print(f"{metric}: F = {f_stat:.3f}, p = {p_val:.4f}")

# === 3.5.2 Sentiment Arc Smoothing and Visualization === #

with open("sentiment_arcs.json", "r") as f:
    sentiment_data = json.load(f)

plt.figure(figsize=(10, 6))
for section, arc in sentiment_data.items():
    smoothed = uniform_filter1d(arc, size=5)
    plt.plot(smoothed, label=section)

plt.title("Smoothed Emotional Arcs by Narrative Section")
plt.xlabel("Window Index")
plt.ylabel("Smoothed Sentiment Score")
plt.legend()
plt.tight_layout()
plt.savefig("smoothed_sentiment_arcs.png")
plt.close()
print("Saved: smoothed_sentiment_arcs.png")

# === 3.5.3 Topic Heatmap and Jensen-Shannon Topic Shift === #

df_topic = pd.read_csv("topic_windows.csv")  # 包含 Topic_0...Topic_4 和 section

heatmap_data = df_topic.groupby('section').mean().iloc[:, :-1]

plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title("Average Topic Distribution per Section")
plt.xlabel("Topic")
plt.ylabel("Narrative Section")
plt.tight_layout()
plt.savefig("topic_heatmap.png")
plt.close()
print("Saved: topic_heatmap.png")

topic_probs = df_topic.iloc[:, :-1].to_numpy()

js_distances = []
for i in range(1, len(topic_probs)):
    js = jensenshannon(topic_probs[i - 1], topic_probs[i])
    js_distances.append(js)

plt.figure(figsize=(10, 4))
plt.plot(js_distances, color='darkorange')
plt.title("Topic Shift Rate (Jensen-Shannon Divergence)")
plt.xlabel("Window Index")
plt.ylabel("JS Divergence")
plt.tight_layout()
plt.savefig("topic_shift_curve.png")
plt.close()
print("Saved: topic_shift_curve.png")

# === 3.5.4 Statistical Tests: ANOVA & Spearman Correlation === #

# 计算情绪波动（标准差）并合并到风格指标表
sentiment_std = {section: np.std(scores) for section, scores in sentiment_data.items()}
df_style['sentiment_std'] = df_style['section'].map(sentiment_std)

print("\n--- Additional ANOVA on sentiment_std ---")
groups = [group['sentiment_std'].values for _, group in df_style.groupby("section")]
f_stat, p_val = f_oneway(*groups)
print(f"Sentiment Std: F = {f_stat:.3f}, p = {p_val:.4f}")

print("\n--- Spearman Correlations (Syntactic Complexity vs Sentiment Volatility) ---")
for metric in ["MSL", "SCR"]:
    corr, p = spearmanr(df_style[metric], df_style["sentiment_std"], nan_policy='omit')
    print(f"{metric} vs Sentiment Std: Spearman's rho = {corr:.3f}, p = {p:.4f}")

# === 3.5.5 Export Comprehensive PDF Report === #

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_page()
pdf.set_font("Times", 'B', 16)
pdf.cell(0, 10, "Faulkner's The Sound and the Fury Text Mining Report", ln=True, align="C")
pdf.ln(10)
pdf.set_font("Times", '', 12)
pdf.multi_cell(0, 10,
               "This report summarizes computational text analysis results, including stylistic features, "
               "emotional arcs, topic modeling, and statistical validations across the narrative sections."
               )

for metric in metrics:
    pdf.add_page()
    pdf.set_font("Times", 'B', 14)
    pdf.cell(0, 10, f"{metric} by Narrative Section", ln=True)
    pdf.image(f"{metric}_by_section.png", w=180)
    pdf.ln(5)
    pdf.set_font("Times", '', 11)
    pdf.multi_cell(0, 10, f"The boxplot shows the distribution of {metric} across the four narrative sections.")

pdf.add_page()
pdf.set_font("Times", 'B', 14)
pdf.cell(0, 10, "Smoothed Emotional Arcs by Narrative Section", ln=True)
pdf.image("smoothed_sentiment_arcs.png", w=180)
pdf.ln(5)
pdf.set_font("Times", '', 11)
pdf.multi_cell(0, 10,
               "This line chart displays the smoothed sentiment scores computed over sliding windows, "
               "highlighting emotional trajectories for each narrator.")

pdf.add_page()
pdf.set_font("Times", 'B', 14)
pdf.cell(0, 10, "Average Topic Distribution per Section", ln=True)
pdf.image("topic_heatmap.png", w=180)
pdf.ln(5)
pdf.set_font("Times", '', 11)
pdf.multi_cell(0, 10,
               "The heatmap visualizes the average proportion of each topic in the different narrative sections.")

pdf.add_page()
pdf.set_font("Times", 'B', 14)
pdf.cell(0, 10, "Topic Shift Rate (Jensen-Shannon Divergence)", ln=True)
pdf.image("topic_shift_curve.png", w=180)
pdf.ln(5)
pdf.set_font("Times", '', 11)
pdf.multi_cell(0, 10,
               "This plot shows the degree of thematic change between adjacent text windows, "
               "with peaks indicating topic shifts in the narrative progression.")

pdf.output("faulkner_analysis_report.pdf")
print("Saved: faulkner_analysis_report.pdf")
