import uuid
from typing import Sequence, List

import chromadb
from chromadb.api.types import Embedding, Metadata

class VectorStore:
    def __init__(self, path: str, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, documents: Sequence[str], embeddings: Sequence[Embedding], metadatas: Sequence[Metadata]) -> None:
        ids = [f"{uuid.uuid4()}_{i}" for i in range(len(documents))]
        self.collection.add(documents=list(documents), embeddings=list(embeddings), metadatas=list(metadatas), ids=ids)

    def query(self, query_embedding: Embedding, k: int) -> List[str]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        docs = results.get("documents")
        return docs[0] if docs and docs[0] else []

    def delete(self, filename: str) -> None:
        self.collection.delete(where={"filename": filename})
