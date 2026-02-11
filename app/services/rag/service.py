from typing import Optional, Dict, Any
import numpy as np
from mistralai import Mistral
from app.core.config import settings
from app.services.rag.vector_store import VectorStore
from app.services.rag.llm import LLMService
from app.services.rag.chunker import chunk_text
from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

class RAGService:
    def __init__(self):
        self.mistral_client = Mistral(api_key=settings.mistral_api_key)
        self.vector_store = VectorStore(path=settings.chroma_db_path)
        self.llm_service = LLMService(client=self.mistral_client, model=settings.mistral_model, max_tokens=settings.mistral_max_tokens)
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-large')

    def get_embedding(self, text: str) -> Embedding:
        return self.embedding_model.encode(text, convert_to_numpy=True)

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
            hypothetical_answer = self.llm_service.generate_hypothetical_answer(question)
            if not hypothetical_answer:
                hypothetical_answer = question

            try:
                hyde_embedding = self.get_embedding(hypothetical_answer)

            except (ValueError, TypeError):
                return "Error: Failed to generate embedding for the hypothetical answer."

            retrieved_chunks = self.vector_store.hybrid_query(question=question, query_embedding=hyde_embedding, k=k*2)

            if not retrieved_chunks:
                return "No relevant context found."

            scores = self.cross_encoder.predict([(question, doc) for doc in retrieved_chunks])
            ranked_chunks = [doc for _, doc in sorted(zip(scores, retrieved_chunks), reverse=True)]
            reranked_chunks = ranked_chunks[:k]

            return self.llm_service.generate_answer_map_reduce(question, reranked_chunks)

    def delete_document(self, filename: str) -> None:
        self.vector_store.delete(filename=filename)

rag_service = RAGService()
