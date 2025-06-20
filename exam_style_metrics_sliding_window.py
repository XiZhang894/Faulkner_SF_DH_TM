import pandas as pd

df = pd.read_csv("style_metrics_sliding_window_full.csv")
for metric in ["MSL", "SCR", "NounRatio", "VerbRatio"]:  # 选几个常见指标
    print(metric, "unique values per section:")
    print(df.groupby("section")[metric].nunique())
    print("-" * 40)
