"""
Task 2: Discipline at the FIFA World Cup 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("task2_cards_data.csv")

df["cards_per_match"] = df["yellow_cards"] / df["matches_played"]

df["group"] = np.where(df["confederation"] == "UEFA", "UEFA", "Non-UEFA")

print("First 5 rows of the wrangled dataset:")
print(df.head(), "\n")

POPULATION = df.copy()
SAMPLE_SIZE = 34
np.random.seed(42)
sample = POPULATION.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f"Population size: {len(POPULATION)} teams")
print(f"Sample size: {len(sample)} teams (simple random sample, seed=42)\n")

uefa_sample = sample.loc[sample["group"] == "UEFA", "cards_per_match"]
non_uefa_sample = sample.loc[sample["group"] == "Non-UEFA", "cards_per_match"]

print(f"UEFA teams in sample: {len(uefa_sample)}")
print(f"Non-UEFA teams in sample: {len(non_uefa_sample)}\n")

desc = sample.groupby("group")["cards_per_match"].agg(["count", "mean", "std", "min", "max"])
print("Descriptive statistics by group:")
print(desc, "\n")

overall_mean = sample["cards_per_match"].mean()
overall_std = sample["cards_per_match"].std()
print(f"Overall sample mean: {overall_mean:.3f} cards/match")
print(f"Overall sample std dev: {overall_std:.3f}\n")

n = len(sample)
mean = sample["cards_per_match"].mean()
sem = stats.sem(sample["cards_per_match"])
ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)

print(f"95% Confidence Interval for mean cards/match (all sampled teams): "
      f"({ci_low:.3f}, {ci_high:.3f})\n")

t_stat, p_value = stats.ttest_ind(uefa_sample, non_uefa_sample, equal_var=False)

print("Two-sample t-test: UEFA vs Non-UEFA cards per match")
print(f"  UEFA mean:     {uefa_sample.mean():.3f}  (n={len(uefa_sample)})")
print(f"  Non-UEFA mean: {non_uefa_sample.mean():.3f}  (n={len(non_uefa_sample)})")
print(f"  t-statistic:   {t_stat:.3f}")
print(f"  p-value:       {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"  Result: p < {alpha} -> statistically significant difference between groups.\n")
else:
    print(f"  Result: p >= {alpha} -> no statistically significant difference between groups.\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sample.boxplot(column="cards_per_match", by="group", ax=axes[0])
axes[0].set_title("Cards per match: UEFA vs Non-UEFA")
axes[0].set_xlabel("")
axes[0].set_ylabel("Yellow cards per match")
plt.suptitle("")

means = sample.groupby("group")["cards_per_match"].mean()
sems = sample.groupby("group")["cards_per_match"].sem()
ci95 = sems * stats.t.ppf(0.975, df=sample.groupby("group").size() - 1)

axes[1].bar(means.index, means.values, yerr=ci95.values, capsize=8, color=["#1f77b4", "#ff7f0e"])
axes[1].set_title("Mean cards per match (95% CI)")
axes[1].set_ylabel("Yellow cards per match")

plt.tight_layout()
plt.savefig("task2_visualisation.png", dpi=150)
print("Saved chart to task2_visualisation.png")
