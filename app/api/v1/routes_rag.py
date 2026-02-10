from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.services.rag_service import rag_service
from app.services.pdf_service import pdf_service

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

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
        markdown_content = pdf_service.process_upload(content, filename)

        rag_service.ingest_text(markdown_content, metadata={"filename": filename, "source": "upload"})

        return {
            "message": "File processed and ingested successfully",
            "filename": filename,
            "content_length": len(markdown_content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_files():
    try:
        files = pdf_service.list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        deleted = pdf_service.delete_file(filename)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

        rag_service.delete_document(filename)

        return {"message": f"File '{filename}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer = rag_service.query(request.question, request.k)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
