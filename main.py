from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

st.title("⚖️ Legal Document Assistant – Answer legal questions from case law or contracts.")

load_dotenv()

loader = PyMuPDFLoader("growwcontractnote.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

chunks = text_splitter.split_documents(docs)

# Storing chunks in vecotre store 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss store")

# Building an RAG chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = db.as_retriever(),
    return_source_documents = True
)
# Streamlit user input
query = st.text_input("Ask a question about the document: ")

# check if Query is not empty
if query:
    legal_prompt = f"You are a legal assistant. Answer the following legal question based only on the provided document. Be precise and avoid assumptions.\n\nQuestion: {query}"
    response = qa_chain.invoke({"query": legal_prompt})
    st.write("📌 **Answer:**", response["result"])
    st.markdown("🔍 **Source Excerpts:**")
    for doc in response["source_documents"]:
        st.markdown(f"```{doc.page_content[:400]}...```")
else: 
    st.write("Please enter a question to get started")
