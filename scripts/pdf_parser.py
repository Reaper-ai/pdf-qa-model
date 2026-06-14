import os
import pdfplumber
from typing import List, Dict, Any

class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    def extract_pages(self) -> List[Dict[Any, Any]]:
        """
        Extracts text page-by-page while maintaining basic structural awareness 
        and capturing essential metadata for downstream citation.
        """
        parsed_pages = []
        
        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # .extract_text layout parameters help manage multi-column flows
                text = page.extract_text(layout=False) or ""
                
                parsed_pages.append({
                    "text": text,
                    "metadata": {
                        "source": self.file_name,
                        "page_number": page_num,
                        "total_pages": len(pdf.pages)
                    }
                })
        return parsed_pages