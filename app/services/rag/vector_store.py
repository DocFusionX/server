import uuid
from typing import Sequence, List

import chromadb
from chromadb.api.types import Embedding, Metadata
from rank_bm25 import BM25Okapi

class VectorStore:
    def __init__(self, path: str, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.documents: List[str] = self.collection.get().get("documents") or []
        self.tokenized_documents = [doc.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def add(self, documents: Sequence[str], embeddings: Sequence[Embedding], metadatas: Sequence[Metadata]) -> None:
        ids = [f"{uuid.uuid4()}_{i}" for i in range(len(documents))]
        self.collection.add(documents=list(documents), embeddings=list(embeddings), metadatas=list(metadatas), ids=ids)
        self.documents.extend(documents)
        self.tokenized_documents.extend([doc.split(" ") for doc in documents])
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def vector_query(self, query_embedding: Embedding, k: int) -> List[str]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        docs = results.get("documents")
        return docs[0] if docs and docs[0] else []

    def keyword_query(self, query: str, k: int) -> List[str]:
        tokenized_query = query.split(" ")
        return self.bm25.get_top_n(tokenized_query, self.documents, n=k)

    def hybrid_query(self, question: str, query_embedding: Embedding, k: int) -> List[str]:
        vector_results = self.vector_query(query_embedding, k)
        keyword_results = self.keyword_query(question, k)

        combined_results = vector_results + keyword_results
        unique_results = list(dict.fromkeys(combined_results))

        return unique_results

    def delete(self, filename: str) -> None:
        self.collection.delete(where={"filename": filename})
