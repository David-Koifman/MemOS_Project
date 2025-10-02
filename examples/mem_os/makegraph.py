#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

df["short_q"] = df["question"].str.slice(0, 30) + "..."

methods = ["sim_memos", "sim_bm25", "sim_combined"]

plt.figure(figsize=(12, 6))
x = range(len(df))

bar_width = 0.25
plt.bar([i - bar_width for i in x], df["sim_memos"], width=bar_width, label="MEMOS")
plt.bar(x, df["sim_bm25"], width=bar_width, label="BM25")
plt.bar([i + bar_width for i in x], df["sim_combined"], width=bar_width, label="Combined")

plt.xticks(x, df["short_q"], rotation=30, ha="right")
plt.ylabel("Similarity (cosine)")
plt.title("Сравнение методов: MEMOS vs BM25 vs Combined")
plt.legend()
plt.tight_layout()


plt.savefig("comparison_results.png", dpi=300)
plt.show()