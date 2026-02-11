from typing import Optional, Dict, Any, List
import numpy as np
from mistralai import Mistral
from app.core.config import settings
from app.services.rag.vector_store import VectorStore
from app.services.rag.llm import LLMService
from app.services.rag.chunker import chunk_text
from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager
from graph_retriever.edges import EdgeSpec
from langchain_core.embeddings import Embeddings

class TransformerEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

class RAGService:
    def __init__(self):
        self.mistral_client = Mistral(api_key=settings.mistral_api_key)
        self.vector_store = VectorStore(path=settings.chroma_db_path)
        self.llm_service = LLMService(client=self.mistral_client, model=settings.mistral_model, max_tokens=settings.mistral_max_tokens)
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.lc_embeddings = TransformerEmbeddings(self.embedding_model)
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-large')

        self.edges: List[EdgeSpec] = [
            ("next_chunk_id", "chunk_id"),
            ("prev_chunk_id", "chunk_id"),
            ("filename", "filename"),
        ]

    def get_embedding(self, text: str) -> Embedding:
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
            chunks_with_meta = chunk_text(text, metadata)

            chunks = [item[0] for item in chunks_with_meta]
            metadatas = [item[1] for item in chunks_with_meta]

            if not chunks:
                return

            embeddings = self.embedding_model.encode(
                chunks,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            embeddings_list = [emb for emb in embeddings]
            self.vector_store.add(documents=chunks, embeddings=embeddings_list, metadatas=metadatas)

    def query(self, question: str, k: int = 10) -> str | None:
            hypothetical_answer = self.llm_service.generate_hypothetical_answer(question)
            if not hypothetical_answer:
                hypothetical_answer = question

            lc_store = self.vector_store.get_langchain_store(self.lc_embeddings)

            retriever = GraphRetriever(
                store=lc_store,
                edges=self.edges,
                strategy=Eager(k=k*2, start_k=k, max_depth=1)
            )

            try:
                retrieved_docs = retriever.invoke(hypothetical_answer)
                retrieved_chunks = [doc.page_content for doc in retrieved_docs]
            except Exception as e:
                print(f"Graph retrieval failed: {e}. Falling back to hybrid query.")
                try:
                    hyde_embedding = self.get_embedding(hypothetical_answer)
                    retrieved_chunks = self.vector_store.hybrid_query(question=question, query_embedding=hyde_embedding, k=k*2)
                except Exception:
                    return "Error during retrieval."

            if not retrieved_chunks:
                return "No relevant context found."

            scores = self.cross_encoder.predict([(question, doc) for doc in retrieved_chunks])
            ranked_chunks = [doc for _, doc in sorted(zip(scores, retrieved_chunks), reverse=True)]
            reranked_chunks = ranked_chunks[:k]

            return self.llm_service.generate_answer_map_reduce(question, reranked_chunks)

    def delete_document(self, filename: str) -> None:
        self.vector_store.delete(filename=filename)

rag_service = RAGService()
