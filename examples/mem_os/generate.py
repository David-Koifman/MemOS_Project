import pandas as pd
from tabulate import tabulate

# Загружаем результаты
df = pd.read_csv("results.csv")

# Оставляем ключевые столбцы для анализа
cols = [
    "question", "gold",
    "memos_ans", "bm25_ans", "faiss_ans", "sentencet_ans", "combined_ans",
    "sim_memos", "sim_bm25", "sim_faiss", "sim_sentencet", "sim_combined",
    "winner"
]

df = df[cols]

# Печатаем сводную таблицу (по каждому вопросу)
table = []
for _, row in df.iterrows():
    table.append([
        row["question"],
        row["winner"],
        row["sim_memos"],
        row["sim_faiss"],
        row["sim_combined"],
        row["sim_bm25"],
        row["sim_sentencet"]
    ])

print("\n📊 Победители по каждому вопросу:\n")
print(tabulate(table, headers=[
    "Вопрос", "Лучший метод", "MEMOS", "FAISS", "Combined", "BM25", "SentT"
], tablefmt="pretty"))

# Дополнительно сохраним полную таблицу в удобный формат
df.to_excel("results_detailed.xlsx", index=False)
print("\n✅ Подробные результаты сохранены в results_detailed.xlsx")