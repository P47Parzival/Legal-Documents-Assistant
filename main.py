from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader("OS-Course Outline.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

chunks = text_splitter.split_documents(docs)
# print(chunks[0].page_content)       #this will print first chunk of text