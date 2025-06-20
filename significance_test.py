import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"

df = pd.read_csv("style_metrics_sliding_window_full.csv")

metrics = [
    "MSL", "SCR", "PassiveAuxRatio", "PastParticipleRatio", "TTR", "AWL",
    "MTLD", "NounRatio", "VerbRatio", "AdjRatio", "AdvRatio",
    "AvgClauseLength", "MeanDependencyDistance", "SubordinationIndex"
]

alpha = 0.05

for metric in metrics:
    print(f"Analyzing {metric}...")

    groups = [group[metric].dropna() for _, group in df.groupby("section")]
    if any(len(g.unique()) == 1 for g in groups):
        print(f"Skipping {metric} due to no variation within at least one section.\n")
        continue

    formula = f"{metric} ~ C(section)"
    model = smf.ols(formula, data=df).fit()
    anova_results = sm.stats.anova_lm(model, typ=1)

    f_val = anova_results["F"].iloc[0]
    p_val = anova_results["PR(>F)"].iloc[0]

    formatted_p_val = "< 1e-10" if p_val == 0 else f"{p_val:.2e}"
    table_data = [["Test", "Group 1", "Group 2", "Mean Diff", "p-value", "Significant?"],
                  ["ANOVA", "-", "-", f"{f_val:.3f}", formatted_p_val, "Yes" if p_val < alpha else "No"]]

    if p_val < alpha:
        tukey = pairwise_tukeyhsd(endog=df[metric], groups=df["section"], alpha=alpha)
        for row in tukey._results_table.data[1:]:
            group1, group2, meandiff, p_adj, lower, upper, reject = row
            formatted_p_adj = "< 1e-10" if p_adj == 0 else f"{p_adj:.2e}"
            table_data.append(["Tukey", group1, group2, f"{meandiff:.3f}", formatted_p_adj, "Yes" if reject else "No"])

    # Plot table
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(table_data)))
    ax.axis("off")
    ax.set_title(f"ANOVA and Tukey HSD Results for {metric}", fontsize=14)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    plt.tight_layout()
    output_filename = f"{metric}_anova_tukey_table.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved result table: {output_filename}\n")
    print("-" * 50)
