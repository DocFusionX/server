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

    structure_summary = extract_structure(text)
    if structure_summary:
        meta = (metadata or {}).copy()
        meta["is_structure"] = True
        chunks_with_meta.append((structure_summary, meta))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        base_meta = (metadata or {}).copy()
        base_meta["chunk_index"] = i
        chunks_with_meta.append((chunk, base_meta))

    return chunks_with_meta
