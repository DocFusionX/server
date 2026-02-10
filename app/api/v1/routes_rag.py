from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.services.rag_service import rag_service

router = APIRouter()

class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    question: str
    k: int = 3

@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    try:
        rag_service.ingest_text(request.text, request.metadata)
        return {"message": "Document ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer = rag_service.query(request.question, request.k)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
