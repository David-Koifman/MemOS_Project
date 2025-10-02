import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from memos.mem_os.main import MOS
from rank_bm25 import BM25Okapi
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer

# CONFIG
BOOK_CHUNKS = "book_chunks.json"

QUESTIONS = [
    "Кто автор книги?",
    "В чем основная цель книги?",
    "Какие четыре принципа переговоров выделяют авторы?",
    "Что такое BATNA?",
    "Чем отличается позиционный торг от переговоров на основе интересов?",
    "Как бороться с нечестными тактиками в переговорах?",
    "Что значит разделять людей и проблему?",
]

# Эталонные ответы
GOLD_ANSWERS = {
    "Кто автор книги?": "Роджер Фишер и Уильям Юри.",
    "В чем основная цель книги?": "Показать, как вести переговоры, ориентируясь на интересы сторон, а не позиции, чтобы находить взаимовыгодные решения.",
    "Какие четыре принципа переговоров выделяют авторы?": "1) Отделять людей от проблемы. 2) Фокусироваться на интересах, а не на позициях. 3) Генерировать множество вариантов решений. 4) Использовать объективные критерии.",
    "Что такое BATNA?": "BATNA — наилучшая альтернатива переговорам. Это резервный вариант, если соглашение не достигнуто.",
    "Чем отличается позиционный торг от переговоров на основе интересов?": "Позиционный торг — это жесткое отстаивание позиций. Переговоры на основе интересов — поиск решений, удовлетворяющих истинные потребности сторон.",
    "Как бороться с нечестными тактиками в переговорах?": "Распознать тактику, открыто обсудить её, использовать объективные критерии и не поддаваться манипуляциям.",
    "Что значит разделять людей и проблему?": "Это означает разграничивать личные отношения участников и суть обсуждаемого вопроса, чтобы избежать конфликтов и сосредоточиться на решении.",
}
# ----------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Init MEMOS
os.environ["MOS_TEXT_MEM_TYPE"] = "general_text"
mos = MOS.simple()
print("✅ MOS initialized")

# 2. Load chunks
with open(BOOK_CHUNKS, "r", encoding="utf-8") as f:
    chunks = json.load(f)

docs = [c["text"] for c in chunks]
print(f"✅ Loaded {len(docs)} chunks")

# 3. BM25
bm25 = BM25Okapi([d.split() for d in docs])
print("✅ BM25 index ready")

# 4. FAISS (OpenAI embeddings)
def get_embedding(text: str):
    if not text:
        return np.zeros(1536)
    emb = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(emb.data[0].embedding)

doc_embeddings = [get_embedding(d) for d in docs]
dim = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings).astype("float32"))
print("✅ FAISS index ready")

# 5. SentenceTransformers
st_model = SentenceTransformer("all-MiniLM-L6-v2")
st_doc_embeddings = st_model.encode(docs, convert_to_numpy=True)
print("✅ SentenceTransformers index ready")

# Search functions
def search_memos(q: str) -> str:
    return mos.chat(q)

def search_bm25(q: str, topk=1) -> str:
    scores = bm25.get_scores(q.split())
    top = sorted(zip(scores, docs), key=lambda x: -x[0])[:topk]
    return "\n---\n".join([d for _, d in top])

def search_faiss(q: str, topk=1) -> str:
    q_emb = get_embedding(q).astype("float32")
    D, I = index.search(np.array([q_emb]), topk)
    return "\n---\n".join([docs[i] for i in I[0]])

def search_sentencet(q: str, topk=1) -> str:
    q_emb = st_model.encode([q], convert_to_numpy=True)
    sims = np.dot(st_doc_embeddings, q_emb[0])
    idx = int(np.argmax(sims))
    return docs[idx]

def search_combined(q: str) -> str:
    memos_ans = search_memos(q)
    faiss_ans = search_faiss(q)
    prompt = f"""
Вопрос: {q}

Ответ 1 (MEMOS): {memos_ans}

Ответ 2 (FAISS): {faiss_ans}

Сравни ответы и составь лучший финальный ответ, максимально точный и связанный с вопросом.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# Similarity util
def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Run queries
results = []
for q in QUESTIONS:
    gold = GOLD_ANSWERS[q]

    ans_memos = search_memos(q)
    ans_bm25 = search_bm25(q)
    ans_faiss = search_faiss(q)
    ans_sentencet = search_sentencet(q)
    ans_combined = search_combined(q)

    gold_emb = get_embedding(gold)
    sim_memos = cosine_sim(get_embedding(ans_memos), gold_emb)
    sim_bm25 = cosine_sim(get_embedding(ans_bm25), gold_emb)
    sim_faiss = cosine_sim(get_embedding(ans_faiss), gold_emb)
    sim_sent = cosine_sim(get_embedding(ans_sentencet), gold_emb)
    sim_comb = cosine_sim(get_embedding(ans_combined), gold_emb)

    results.append({
        "question": q,
        "gold": gold,
        "memos_ans": ans_memos,
        "bm25_ans": ans_bm25,
        "faiss_ans": ans_faiss,
        "sentencet_ans": ans_sentencet,
        "combined_ans": ans_combined,
        "sim_memos": round(sim_memos, 3),
        "sim_bm25": round(sim_bm25, 3),
        "sim_faiss": round(sim_faiss, 3),
        "sim_sentencet": round(sim_sent, 3),
        "sim_combined": round(sim_comb, 3),
        "winner": max(
            [
                ("MEMOS", sim_memos),
                ("BM25", sim_bm25),
                ("FAISS", sim_faiss),
                ("SentenceT", sim_sent),
                ("Combined", sim_comb),
            ],
            key=lambda x: x[1]
        )[0],
    })

#Save to CSV
with open("results.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("✅ Results saved to results.csv")

#Graph
labels = [r["question"] for r in results]
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - 2*width, [r["sim_memos"] for r in results], width, label="MEMOS")
ax.bar(x - width, [r["sim_bm25"] for r in results], width, label="BM25")
ax.bar(x, [r["sim_faiss"] for r in results], width, label="FAISS")
ax.bar(x + width, [r["sim_sentencet"] for r in results], width, label="SentenceT")
ax.bar(x + 2*width, [r["sim_combined"] for r in results], width, label="Combined")

ax.set_ylabel("Схожесть с эталоном")
ax.set_title("Сравнение методов поиска")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.legend()

plt.tight_layout()
plt.savefig("results.png", dpi=200)
plt.show()