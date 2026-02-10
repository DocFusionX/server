import chromadb
import uuid
import re
from typing import List, Optional, Dict, Any
from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage
from app.core.config import settings

class RAGService:
    def __init__(self):
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="documents")

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model="mistral-embed", inputs=[text])
        if response.data:
            embedding = response.data[0].embedding
            if isinstance(embedding, list):
                return embedding
        return []

    def extract_structure(self, text: str) -> str:
        headers = re.findall(r'^(#{1,6}\s+.+)$', text, re.MULTILINE)
        if not headers:
            return ""
        return "Document Structure and Table of Contents:\n" + "\n".join(headers)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        chunk_size = 4000
        overlap = 200

        chunks = []

        structure_summary = self.extract_structure(text)
        if structure_summary:
            chunks.append(structure_summary)

        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            if not embedding:
                continue

            chunk_metadata = (metadata or {}).copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["is_structure"] = (i == 0 and structure_summary != "" and chunk == structure_summary)

            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[chunk_metadata],
                ids=[f"{str(uuid.uuid4())}_{i}"]
            )

    def query(self, question: str, k: int = 3) -> str | None:
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            return "Error: Failed to generate embedding for the question."

        results = self.collection.query(query_embeddings=[question_embedding], n_results=k)

        if not results or not results['documents'] or not results['documents'][0]:
            return "No relevant context found."

        context = "\n".join(results['documents'][0])

        messages = [
            SystemMessage(content="You are a helpful assistant. Answer the question based on the context provided."),
            UserMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]

        chat_response = self.client.chat.complete(
            model=settings.mistral_model,
            messages=messages,
            max_tokens=settings.mistral_max_tokens
        )

        if chat_response.choices:
            content = chat_response.choices[0].message.content
            if isinstance(content, str):
                return content

        return None

    def delete_document(self, filename: str) -> None:
        self.collection.delete(where={"filename": filename})

rag_service = RAGService()
