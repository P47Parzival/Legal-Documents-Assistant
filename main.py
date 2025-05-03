from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import retrieval_qa
from langchain_google_genai import ChatGoogleGenerativeAI

loader = PyMuPDFLoader("OS-Course Outline.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

chunks = text_splitter.split_documents(docs)

# Storing chunks in vecotre store 
embeddings = GooglePalmEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss store")

# Building an RAG chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
qa_chain = retrieval_qa.from_chain_type(
    llm = llm,
    retriever = db.as_retriever(),
    return_source_documents = True
)
query = "What is this documnet about?"
response = qa_chain({"query": query})
print(response["result"])

# print(chunks[0].page_content)       #this will print first chunk of text