import re
from typing import List, Tuple, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_structure(text: str) -> str:
    headers = re.findall(r'^(#{1,6}\s+.+)', text, re.MULTILINE)
    if not headers:
        return ""
    return "Document Structure and Table of Contents:\n" + "\n".join(headers)

def chunk_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    chunks_with_meta = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    structure_summary = extract_structure(text)
    if structure_summary:
        structure_chunks = text_splitter.split_text(structure_summary)
        for i, chunk in enumerate(structure_chunks):
            meta = (metadata or {}).copy()
            meta["is_structure"] = True
            meta["chunk_index"] = i
            chunks_with_meta.append((chunk, meta))

    body_chunks = text_splitter.split_text(text)

    chunk_offset = len(chunks_with_meta)
    filename = (metadata or {}).get("filename", "unknown")

    for i, chunk in enumerate(body_chunks):
        base_meta = (metadata or {}).copy()
        current_idx = chunk_offset + i
        base_meta["chunk_index"] = current_idx
        base_meta["chunk_id"] = f"{filename}_{current_idx}"

        if i < len(body_chunks) - 1:
            base_meta["next_chunk_id"] = f"{filename}_{current_idx + 1}"
        if i > 0:
            base_meta["prev_chunk_id"] = f"{filename}_{current_idx - 1}"

        chunks_with_meta.append((chunk, base_meta))

    return chunks_with_meta
