import chromadb
import uuid
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
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        if response.data:
            embedding = response.data[0].embedding
            if isinstance(embedding, list):
                return embedding
        return []

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        embedding = self.get_embedding(text)
        if not embedding:
            raise ValueError("Failed to generate embedding")

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())]
        )

    def query(self, question: str, k: int = 3) -> str | None:
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            return "Error: Failed to generate embedding for the question."

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=k
        )

        if not results or not results['documents'] or not results['documents'][0]:
            return "No relevant context found."

        context = "\n".join(results['documents'][0])

        messages = [
            SystemMessage(content="You are a helpful assistant. Answer the question based on the context provided."),
            UserMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]

        chat_response = self.client.chat.complete(
            model="mistral-tiny",
            messages=messages
        )

        if chat_response.choices:
            content = chat_response.choices[0].message.content
            if isinstance(content, str):
                return content

        return None

rag_service = RAGService()
