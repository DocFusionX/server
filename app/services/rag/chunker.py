import re
from typing import List, Tuple, Dict, Any, Optional

def extract_structure(text: str) -> str:
    headers = re.findall(r'^(#{1,6}\s+.+)', text, re.MULTILINE)
    if not headers:
        return ""
    return "Document Structure and Table of Contents:\n" + "\n".join(headers)

def chunk_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    chunks_with_meta = []

    structure_summary = extract_structure(text)
    if structure_summary:
        meta = (metadata or {}).copy()
        meta["is_structure"] = True
        chunks_with_meta.append((structure_summary, meta))

    semantic_chunks = re.split(r'^(?=#{1,6}\s)', text, flags=re.MULTILINE)

    max_chunk_size = 2000
    overlap = 200

    for i, semantic_chunk in enumerate(semantic_chunks):
        semantic_chunk = semantic_chunk.strip()
        if not semantic_chunk:
            continue

        header_match = re.match(r'^(#{1,6}\s+.+)', semantic_chunk)
        header = header_match.group(1).strip() if header_match else None

        base_meta = (metadata or {}).copy()
        if header:
            base_meta["header"] = header
        base_meta["chunk_index"] = i

        if len(semantic_chunk) > max_chunk_size:
            start = 0
            while start < len(semantic_chunk):
                end = start + max_chunk_size
                chunk_part = semantic_chunk[start:end]
                chunks_with_meta.append((chunk_part, base_meta.copy()))
                start += max_chunk_size - overlap
        else:
            chunks_with_meta.append((semantic_chunk, base_meta.copy()))

    return chunks_with_meta
