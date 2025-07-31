import easyocr
import numpy as np
from PIL import Image
import pymupdf
import io


reader = easyocr.Reader(['en'], gpu=False)

def parse_pdf(file: io.BytesIO, is_ocr: bool = False) -> str:
    """
    Parses a PDF file and extracts text, with optional OCR for scanned PDFs.

    :param file: PDF file as BytesIO
    :param is_ocr: Whether to use OCR for scanned PDFs
    :return: Extracted text as a string
    """
    try:
        document = pymupdf.open(stream=file, filetype='pdf')
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    content = []

    for page in document:
        if is_ocr:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                results = reader.readtext(np.array(img), detail=0)
                text = " ".join(results)
            except Exception as e:
                text = f"[ERROR performing OCR on page: {e}]"
        else:
            try:
                text = page.get_text()
            except Exception as e:
                text = f"[ERROR extracting text from page: {e}]"

        content.append(text)

    return "\n".join(content)
