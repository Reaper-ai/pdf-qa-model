from pdf_parser import PDFParser
from cleaner import TextNormalizer
from chunker import SemanticChunker

def test_pdf_parser():
    parser = PDFParser("sample.pdf")
    pages = parser.extract_pages()
    assert len(pages) > 0, "No pages extracted from PDF"
    print(f"Extracted {len(pages)} pages successfully.")
    print("Sample page metadata:", pages[0]["metadata"])
    print("Sample page text snippet:", pages[0]["text"][:100])

test_pdf_parser()