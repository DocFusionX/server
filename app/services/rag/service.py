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

    def query(self, question: str, k: int = 5) -> str | None:
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

                if not retrieved_docs:
                    return "No relevant context found."

                doc_texts = [doc.page_content for doc in retrieved_docs]
                scores = self.cross_encoder.predict([(question, text) for text in doc_texts])

                scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
                reranked_docs = [doc for score, doc in scored_docs[:k]]

                formatted_contexts = []
                seen_structures = set()
                global_structure_context = ""

                for doc in reranked_docs:
                    filename = doc.metadata.get("filename", "Unknown Source")

                    structure = doc.metadata.get("document_structure")

                    if doc.metadata.get("is_structure") and not structure:
                         structure = doc.page_content

                    if structure and filename not in seen_structures:
                        global_structure_context += f"Document Structure for {filename}:\n{structure}\n\n"
                        seen_structures.add(filename)

                    if doc.metadata.get("is_structure"):
                        header = "[DOCUMENT STRUCTURE / TABLE OF CONTENTS]"
                    else:
                        idx = doc.metadata.get("chunk_index", "N/A")
                        header = f"[Source: {filename}, Segment: {idx}]"

                    formatted_contexts.append(f"{header}\n{doc.page_content}")

                if global_structure_context:
                    formatted_contexts.insert(0, f"--- GLOBAL CONTEXT ---\n{global_structure_context}--- END GLOBAL CONTEXT ---")

                return self.llm_service.generate_answer(question, formatted_contexts)

            except Exception as e:
                print(f"Retrieval or generation failed: {e}")
                return "Error during retrieval or answer generation."

    def delete_document(self, filename: str) -> None:
        self.vector_store.delete(filename=filename)

    def clear_database(self) -> None:
        self.vector_store.clear()

rag_service = RAGService()
