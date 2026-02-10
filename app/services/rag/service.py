from typing import Optional, Dict, Any
import numpy as np
from mistralai import Mistral
from app.core.config import settings
from app.services.rag.vector_store import VectorStore
from app.services.rag.llm import LLMService
from app.services.rag.chunker import chunk_text
from chromadb.api.types import Embedding

class RAGService:
    def __init__(self):
        self.mistral_client = Mistral(api_key=settings.mistral_api_key)
        self.vector_store = VectorStore(path=settings.chroma_db_path)
        self.llm_service = LLMService(client=self.mistral_client, model=settings.mistral_model, max_tokens=settings.mistral_max_tokens)

    def get_embedding(self, text: str) -> Embedding:
        response = self.mistral_client.embeddings.create(model="mistral-embed", inputs=[text])

        if not response.data:
            raise ValueError("No embedding returned")

        embedding = response.data[0].embedding
        if not isinstance(embedding, list):
            raise TypeError("Embedding must be a list of floats")

        return np.array(embedding)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
            chunks_with_meta = chunk_text(text, metadata)

            chunks = [item[0] for item in chunks_with_meta]

            metadatas = [item[1] for item in chunks_with_meta]

            valid_embeddings = []
            valid_chunks = []
            valid_metadatas = []

            for chunk, meta in zip(chunks, metadatas):
                try:
                    embedding = self.get_embedding(chunk)
                    valid_embeddings.append(embedding)
                    valid_chunks.append(chunk)
                    valid_metadatas.append(meta)

                except (ValueError, TypeError):
                    continue

            if not valid_embeddings:
                return

            self.vector_store.add(documents=valid_chunks, embeddings=valid_embeddings, metadatas=valid_metadatas)

    def query(self, question: str, k: int = 10) -> str | None:
            try:
                question_embedding = self.get_embedding(question)

            except (ValueError, TypeError):
                return "Error: Failed to generate embedding for the question."

            retrieved_chunks = self.vector_store.query(question_embedding, k=k)

            if not retrieved_chunks:
                return "No relevant context found."

            refined_answer = self.llm_service.get_initial_answer(question, retrieved_chunks[0])
            if refined_answer is None:
                return "Could not generate an initial answer."

            for i in range(1, len(retrieved_chunks)):
                refined_answer = self.llm_service.get_refined_answer(question=question, existing_answer=refined_answer, new_context=retrieved_chunks[i])
                if refined_answer is None:
                    return f"Failed to refine the answer at step {i}"
            return refined_answer

    def delete_document(self, filename: str) -> None:
        self.vector_store.delete(filename=filename)

rag_service = RAGService()
