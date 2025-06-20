import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perform_anova_and_tukey(df, metric_col, group_col="section", alpha=0.05):
    """
    Perform one-way ANOVA and Tukey HSD post-hoc test for a given metric grouped by categories.

    Parameters:
    - df: pandas DataFrame containing data
    - metric_col: string, the column name of the metric to analyze
    - group_col: string, the column name to group by (default 'section')
    - alpha: significance level for tests (default 0.05)

    Returns:
    - None, prints test results
    """
    # Group data by the specified category
    groups = [group[metric_col].dropna() for _, group in df.groupby(group_col)]

    # Skip if any group has no variance (all values identical)
    if any(len(g.unique()) == 1 for g in groups):
        print(f"Skipping {metric_col} due to no variation within at least one group.")
        return

    # Fit OLS model for ANOVA
    model = sm.OLS(df[metric_col], pd.get_dummies(df[group_col])).fit()
    anova_results = sm.stats.anova_lm(model, typ=1)
    f_val = anova_results["F"][0]
    p_val = anova_results["PR(>F)"][0]

    print(f"ANOVA result for {metric_col}: F = {f_val:.3f}, p = {p_val:.4f}")

    # Perform Tukey HSD post-hoc test if ANOVA significant
    if p_val < alpha:
        print(f"Tukey HSD post-hoc test for {metric_col}:")
        tukey = pairwise_tukeyhsd(endog=df[metric_col], groups=df[group_col], alpha=alpha)
        print(tukey.summary())

    print("-" * 50)


# Example of usage:
if __name__ == "__main__":
    df = pd.read_csv("style_metrics_sliding_window.csv")
    metrics = ["MSL", "SCR", "PassiveAuxRatio", "PastParticipleRatio", "TTR", "AWL",
               "MTLD", "NounRatio", "VerbRatio", "AdjRatio", "AdvRatio",
               "AvgClauseLength", "MeanDependencyDistance", "SubordinationIndex"]

    for metric in metrics:
        perform_anova_and_tukey(df, metric)
