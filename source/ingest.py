import pdfplumber
from typing import List


def load_pdf_text(path: str) -> str:
    """Extract text from a PDF file using pdfplumber.

    Returns the concatenated text of all pages.
    """
    texts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
    return "\n\n".join(texts)


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple sliding-window splitter.

    chunk_size: approximate size of each chunk (characters)
    overlap: how many characters to overlap between consecutive chunks
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= text_len:
            break
    return chunks
