from importlib import import_module

documents = import_module("01_documents").documents
preprocess_text = import_module("02_preprocessing").preprocess_text


def chunk_text(text, chunk_size=120, overlap=30):
    """
    Chunks English text into overlapping word blocks.
    Increased default chunk_size and overlap to capture detailed context for RAG.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def build_chunks():
    rows = []

    for document in documents:
        # Pass adjusted chunking parameters for longer, more detailed context
        for chunk_number, chunk in enumerate(chunk_text(document["text"], chunk_size=120, overlap=30)):
            
            # Enrich search_text by including 'category' alongside 'title' and 'chunk'
            combined_search_content = f"{document['title']} {document['category']} {chunk}"
            
            rows.append(
                {
                    "chunk_id": f"{document['id']}_{chunk_number}",
                    "document_id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "is_current": document["is_current"],
                    "chunk_text": chunk,
                    "search_text": preprocess_text(combined_search_content),
                }
            )

    return rows
