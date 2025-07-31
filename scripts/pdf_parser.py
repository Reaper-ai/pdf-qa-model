from io import BytesIO
import pymupdf
from PIL import Image
import pytesseract
import io

def parse_pdf(file: BytesIO, is_ocr: bool = False) -> str:
    """
    parses PDF files

    :param path: path to PDF file
    :param is_ocr: flag whether the PDF file is a scanned image or not
    :return:  PDF content (list of strings per page)
    """
    try:
        document = pymupdf.open(stream=file, filetype='pdf')
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    content = []
    for page in document:
        if is_ocr:
            # Convert page to image
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
            except Exception as e:
                text = f"[ERROR performing OCR on page: {e}]"
        else:
            try:
                text = page.get_text()
            except Exception as e:
                text = f"[ERROR extracting text from page : {e}]"

        content.append(text)

    return '\n'.join(content)
