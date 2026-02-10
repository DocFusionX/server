from pathlib import Path
from docling.document_converter import DocumentConverter
import tempfile

class PDFService:
    def __init__(self):
        self.converter = DocumentConverter()

    def convert_to_markdown(self, file_path: Path) -> str:
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()

    def process_upload(self, file_content: bytes, filename: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / filename
            with open(temp_path, "wb") as f:
                f.write(file_content)

            return self.convert_to_markdown(temp_path)

pdf_service = PDFService()
