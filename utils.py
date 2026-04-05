from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,     # optimized for resumes
        chunk_overlap=50
    )
    return splitter.split_text(text)
