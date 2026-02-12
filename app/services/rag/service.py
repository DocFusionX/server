from typing import Optional, Dict, Any, List
import numpy as np
from mistralai import Mistral
from app.core.config import settings
from app.services.rag.vector_store import VectorStore
from app.services.rag.llm import LLMService
from app.services.rag.chunker import chunk_text
from app.services.rag.integrity import validator
from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

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

    def get_embedding(self, text: str) -> Embedding:
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        filename = (metadata or {}).get("filename", "Unknown")
        issues = validator.validate(text)

        if issues:
            print(f"--- Document Integrity Report for {filename} ---")
            for issue in issues:
                print(f"[{issue.issue_type.upper()}] L{issue.level} '{issue.header}': {issue.message}")
            print("-------------------------------------------------")
        else:
            print(f"--- Document Integrity Check Passed for {filename} ---")

        chunks_with_meta = chunk_text(text, metadata)
        if not chunks_with_meta:
            return

        chunks = [item[0] for item in chunks_with_meta]
        metadatas = [item[1] for item in chunks_with_meta]
        embeddings = self.embedding_model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.vector_store.add(documents=chunks, embeddings=list(embeddings), metadatas=metadatas)

    def _get_neighbors(self, doc: Document) -> List[Document]:
        neighbors = []
        next_id = doc.metadata.get("next_chunk_id")
        prev_id = doc.metadata.get("prev_chunk_id")
        section_id = doc.metadata.get("section_id")

        ids_to_fetch = [i for i in [next_id, prev_id] if i]

        if ids_to_fetch:
            results = self.vector_store.collection.get(ids=ids_to_fetch)
            if results and results.get("ids") and results.get("documents") and results.get("metadatas"):
                docs = results["documents"]
                metas = results["metadatas"]
                if docs and metas:
                    for i in range(len(results["ids"])):
                        neighbors.append(Document(
                            page_content=docs[i],
                            metadata=metas[i] or {}
                        ))

        if section_id and section_id != "structure":
             section_results = self.vector_store.collection.get(where={"section_id": section_id})
             if section_results and section_results.get("ids") and section_results.get("documents") and section_results.get("metadatas"):
                sec_ids = section_results["ids"]
                sec_docs = section_results["documents"]
                sec_metas = section_results["metadatas"]

                if sec_ids and sec_docs and sec_metas:
                    for i in range(len(sec_ids)):
                        d_id = sec_ids[i]
                        if d_id not in ids_to_fetch and d_id != doc.metadata.get("chunk_id"):
                            neighbors.append(Document(
                                page_content=sec_docs[i],
                                metadata=sec_metas[i] or {}
                            ))

        return neighbors

    def query(self, question: str, k: int = 5) -> str | None:
        try:
            hypo_answer = self.llm_service.generate_hypothetical_answer(question) or question
            lc_store = self.vector_store.get_langchain_store(self.lc_embeddings)

            initial_docs = lc_store.similarity_search(hypo_answer, k=k)
            if not initial_docs:
                return "No relevant context found."

            all_docs = []
            seen_ids = set()

            for d in initial_docs:
                chunk_id = d.metadata.get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    all_docs.append(d)
                    seen_ids.add(chunk_id)

                for neighbor in self._get_neighbors(d):
                    n_id = neighbor.metadata.get("chunk_id")
                    if n_id and n_id not in seen_ids:
                         all_docs.append(neighbor)
                         seen_ids.add(n_id)

            if not all_docs:
                return "No relevant context found."

            doc_texts = [d.page_content for d in all_docs]
            scores = self.cross_encoder.predict([(question, t) for t in doc_texts])
            scored_docs = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)

            reranked_docs = [d for s, d in scored_docs[:k*2]]

            formatted_contexts = []
            seen_structures = set()
            global_structure_context = ""

            for doc in reranked_docs:
                filename = doc.metadata.get("filename", "Unknown")
                structure = doc.metadata.get("document_structure")

                if doc.metadata.get("is_structure") and not structure:
                    structure = doc.page_content

                if structure and filename not in seen_structures:
                    global_structure_context += f"Document Structure for {filename}:\n{structure}\n\n"
                    seen_structures.add(filename)

                header = "[STRUCTURE]" if doc.metadata.get("is_structure") else f"[Source: {filename}, Section: {doc.metadata.get('header', 'N/A')}]"
                formatted_contexts.append(f"{header}\n{doc.page_content}")

            if global_structure_context:
                formatted_contexts.insert(0, f"--- GLOBAL CONTEXT ---\n{global_structure_context}--- END GLOBAL CONTEXT ---")

            return self.llm_service.generate_answer(question, formatted_contexts)

        except Exception as e:
            print(f"RAG Error: {e}")
            return "Error during retrieval or answer generation."

    def delete_document(self, filename: str) -> None:
        self.vector_store.delete(filename=filename)

    def clear_database(self) -> None:
        self.vector_store.clear()

rag_service = RAGService()
