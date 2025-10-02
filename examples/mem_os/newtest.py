#!/usr/bin/env python3
"""
Ask questions to already ingested book in MEMOS
"""

from memos.mem_os.main import MOS

# 1. Подключаемся к MEMOS (он уже содержит загруженные чанки из ingest_book.py)
mos = MOS.simple()
print("✅ Connected to MEMOS")

# 2. Задаем вопросы
questions = [
    "Кто автор книги?",
    "В чем основная идея книги?",
    "Что значит отделять людей от проблемы?",
    "Как описан пример переговоров в магазине?",
    "Что советуют делать, если вторая сторона сильнее?",
]

for q in questions:
    print("\n  You:", q)
    print("Bot:", mos.chat(q))