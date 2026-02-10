from typing import List, Optional
from pathlib import Path
from docling.document_converter import DocumentConverter
import os
from app.core.config import settings

class PDFService:
    def __init__(self):
        self.converter = DocumentConverter()
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_markdown(self, file_path: Path) -> str:
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()

    def process_upload(self, file_content: bytes, filename: Optional[str] = None) -> str:
        safe_filename = filename or "document.pdf"
        file_path = self.upload_dir / safe_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        return self.convert_to_markdown(file_path)

    def list_files(self) -> List[str]:
        return [f.name for f in self.upload_dir.glob("*.pdf")]

    def delete_file(self, filename: str) -> bool:
        file_path = self.upload_dir / filename
        if file_path.exists():
            os.remove(file_path)
            return True
        return False

pdf_service = PDFService()
