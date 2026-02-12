import re
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_structure(text: str) -> str:
    headers = re.findall(r'^(#{1,6}\s+.+)', text, re.MULTILINE)
    if not headers:
        return ""
    return "Document Structure and Table of Contents:\n" + "\n".join(headers)

def get_section_id(filename: str, header: str, index: int) -> str:
    unique_str = f"{filename}_{index}_{header}"
    return hashlib.md5(unique_str.encode()).hexdigest()

def chunk_text(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    chunks_with_meta = []
    filename = (metadata or {}).get("filename", "unknown")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False,
    )

    structure_summary = extract_structure(text)

    lines = text.split('\n')
    current_header = "root"

    sections = []
    current_section_lines = []

    for line in lines:
        if re.match(r'^(#{1,6}\s+.+)', line):
            if current_section_lines:
                sections.append((current_header, "\n".join(current_section_lines)))
            current_header = line.strip()
            current_section_lines = [line]
        else:
            current_section_lines.append(line)
    if current_section_lines:
        sections.append((current_header, "\n".join(current_section_lines)))

    if structure_summary:
        structure_chunks = text_splitter.split_text(structure_summary)
        for i, chunk in enumerate(structure_chunks):
            meta = (metadata or {}).copy()
            meta["is_structure"] = True
            meta["chunk_index"] = i
            meta["chunk_id"] = f"{filename}_structure_{i}"
            meta["section_id"] = "structure"
            chunks_with_meta.append((chunk, meta))

    chunk_offset = len(chunks_with_meta)
    global_idx = 0

    for section_idx, (header, section_text) in enumerate(sections):
        section_id = get_section_id(filename, header, section_idx)
        section_chunks = text_splitter.split_text(section_text)

        for i, chunk in enumerate(section_chunks):
            base_meta = (metadata or {}).copy()
            current_idx = chunk_offset + global_idx

            base_meta["chunk_index"] = current_idx
            base_meta["chunk_id"] = f"{filename}_{current_idx}"
            base_meta["section_id"] = section_id
            base_meta["header"] = header

            if structure_summary:
                base_meta["document_structure"] = structure_summary

            chunks_with_meta.append((chunk, base_meta))
            global_idx += 1

    for i in range(len(chunks_with_meta)):
        if chunks_with_meta[i][1].get("is_structure"):
            continue

        if i < len(chunks_with_meta) - 1 and not chunks_with_meta[i+1][1].get("is_structure"):
             chunks_with_meta[i][1]["next_chunk_id"] = chunks_with_meta[i+1][1]["chunk_id"]

        if i > 0 and not chunks_with_meta[i-1][1].get("is_structure"):
             chunks_with_meta[i][1]["prev_chunk_id"] = chunks_with_meta[i-1][1]["chunk_id"]

    return chunks_with_meta
