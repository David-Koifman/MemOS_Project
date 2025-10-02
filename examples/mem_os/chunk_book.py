import os
import re
from pathlib import Path

import fitz
import re
import json

import fitz
from pdf2image import convert_from_path
import pytesseract


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from PDF.
    If page has no text (likely scanned), fall back to OCR.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")

        if not text.strip():
            print(f"[OCR] Page {i+1}")
            images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
            if images:
                text = pytesseract.image_to_string(images[0], lang="eng")

        all_text.append((i + 1, text.strip()))

    return all_text


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits text into chunks with overlap (in words).
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

def main():
    pdf_path = "fisher.pdf"
    output_path = "book_chunks.json"

    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages")

    all_chunks = []
    for page_num, text in pages:
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        for c in chunks:
            all_chunks.append({"page": page_num, "text": c})

    print(f"Total chunks: {len(all_chunks)}")

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunks saved to {output_path}")

    # Preview
    for i, ch in enumerate(all_chunks[:5]):
        print(f"\n--- Chunk {i+1} (page {ch['page']}) ---")
        print(ch['text'][:400], "...")


if __name__ == "__main__":
    main()