"""
Task 3: Possession and knockout progress at the FIFA World Cup 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("task3_possession_data.csv")
print("First 5 rows of the wrangled dataset:")
print(df.head(), "\n")

POPULATION = df.copy()
SAMPLE_SIZE = 24
np.random.seed(7)
sample = POPULATION.sample(n=SAMPLE_SIZE, random_state=7).reset_index(drop=True)

print(f"Population size: {len(POPULATION)} knockout-stage teams")
print(f"Sample size: {len(sample)} teams (simple random sample, seed=7)\n")

qf_sample = sample.loc[sample["progress"] == "Quarterfinal+", "avg_possession_pct"]
early_sample = sample.loc[sample["progress"] == "Early exit", "avg_possession_pct"]

print(f"Quarterfinal+ teams in sample: {len(qf_sample)}")
print(f"Early-exit teams in sample: {len(early_sample)}\n")

desc = sample.groupby("progress")["avg_possession_pct"].agg(["count", "mean", "std", "min", "max"])
print("Descriptive statistics by group:")
print(desc, "\n")

n = len(sample)
mean = sample["avg_possession_pct"].mean()
sem = stats.sem(sample["avg_possession_pct"])
ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)

print(f"95% Confidence Interval for mean possession % (all sampled teams): "
      f"({ci_low:.2f}%, {ci_high:.2f}%)\n")

t_stat, p_value = stats.ttest_ind(qf_sample, early_sample, equal_var=False)

print("Two-sample t-test: Quarterfinal+ vs Early-exit possession")
print(f"  Quarterfinal+ mean: {qf_sample.mean():.2f}%  (n={len(qf_sample)})")
print(f"  Early-exit mean:    {early_sample.mean():.2f}%  (n={len(early_sample)})")
print(f"  t-statistic:        {t_stat:.3f}")
print(f"  p-value:             {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"  Result: p < {alpha} -> statistically significant difference between groups.\n")
else:
    print(f"  Result: p >= {alpha} -> no statistically significant difference between groups.\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sample.boxplot(column="avg_possession_pct", by="progress", ax=axes[0])
axes[0].set_title("Possession %: Quarterfinal+ vs Early exit")
axes[0].set_xlabel("")
axes[0].set_ylabel("Average possession (%)")
plt.suptitle("")

means = sample.groupby("progress")["avg_possession_pct"].mean()
sems = sample.groupby("progress")["avg_possession_pct"].sem()
ci95 = sems * stats.t.ppf(0.975, df=sample.groupby("progress").size() - 1)

axes[1].bar(means.index, means.values, yerr=ci95.values, capsize=8, color=["#2ca02c", "#d62728"])
axes[1].set_title("Mean possession % (95% CI)")
axes[1].set_ylabel("Average possession (%)")

plt.tight_layout()
plt.savefig("task3_visualisation.png", dpi=150)
print("Saved chart to task3_visualisation.png")
