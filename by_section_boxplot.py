import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load the sliding window style metrics data
df = pd.read_csv("style_metrics_sliding_window_full.csv")

# List all numeric style metrics to analyze (excluding 'section' and 'window_start')
metrics = [
    "MSL", "SCR", "PassiveAuxRatio", "PastParticipleRatio", "TTR", "AWL", "MTLD",
    "NounRatio", "VerbRatio", "AdjRatio", "AdvRatio",
    "AvgClauseLength", "MeanDependencyDistance", "SubordinationIndex"
]

print("Descriptive statistics by section:")
print(df.groupby("section")[metrics].describe())

# Perform one-way ANOVA for each metric across sections
print("\n--- ANOVA Results ---")
for metric in metrics:
    groups = [group[metric].dropna().values for name, group in df.groupby("section")]
    try:
        f_val, p_val = stats.f_oneway(*groups)
        print(f"{metric}: F = {f_val:.3f}, p = {p_val:.4f}")
    except Exception as e:
        print(f"{metric}: ANOVA failed - {e}")

# Visualization: Boxplots of each metric by section
for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="section", y=metric, data=df, palette="Set2")
    plt.title(f"{metric} distribution by Section")
    plt.xlabel("Section")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{metric}_by_section_boxplot.png")
    plt.close()
    print(f"Saved plot: {metric}_by_section_boxplot.png")
