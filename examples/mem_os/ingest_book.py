import os
import json
import shutil
from memos.mem_os.main import MOS

#1 Очистка старой памяти
qdrant_path = os.path.join(os.getcwd(), ".memos", "qdrant")
if os.path.exists(qdrant_path):
    print("🗑 Очищаю старую базу Qdrant...")
    shutil.rmtree(qdrant_path)

#2. Конфиг и инициализация MOS
os.environ["MOS_TEXT_MEM_TYPE"] = "general_text"
mos = MOS.simple()
print("✅ MOS initialized")

#3 Загружаем книгу
with open("book_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"✅ Loaded {len(chunks)} chunks")

#4 Добавляем в память с уникальными ID
for i, chunk in enumerate(chunks):
    mos.add(
        memory_content=chunk["text"],
        metadata={"chunk_id": f"chunk_{i+1}", "page": chunk.get("page", None)}
    )
print(f"✅ Added {len(chunks)} chunks into MEMOS memory")

#5 Тестовые запросы =
queries = [
    "Кто автор книги?",
    "О чем эта книга?",
    "Как в ней описываются переговоры в магазине?",
]

for q in queries:
    print("\n👤 You:", q)
    print("🤖 Bot:", mos.chat(q))