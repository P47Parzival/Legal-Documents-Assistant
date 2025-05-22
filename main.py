# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from crawl4ai import AsyncWebCrawler
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import streamlit as st
# import os, asyncio
# import tempfile

# st.title("⚖️ Legal Document Assistant – Answer legal questions from case law, contracts or websites.")

# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("GOOGLE_API_KEY is not set in the .env file.")

# # uploading custom pdf file
# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_file_path = tmp_file.name
#     loader = PyMuPDFLoader(tmp_file_path)
#     docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size = 500,
    # chunk_overlap = 50,
    # )
    # chunks = text_splitter.split_documents(docs)
    # # Storing chunks in vecotre store 
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # db = FAISS.from_documents(chunks, embeddings)
    # db.save_local("faiss store")

    # # Building an RAG chain
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm = llm,
    #     retriever = db.as_retriever(),
    #     return_source_documents = True
    # )
    # # Streamlit user input
    # query = st.text_input("Ask a question about the document: ")

    # # check if Query is not empty
    # if query:
    #     legal_prompt = f"You are a legal assistant. Answer the following legal question based only on the provided document. Be precise and avoid assumptions.\n\nQuestion: {query}"
    #     response = qa_chain.invoke({"query": legal_prompt})
    #     st.write("📌 **Answer:**", response["result"])
    #     st.markdown("🔍 **Source Excerpts:**")
    #     for doc in response["source_documents"]:
    #         st.markdown(f"```{doc.page_content[:400]}...```")
    # else: 
    #     st.write("Please enter a question to get started")
# else:
#     st.warning("Please upload a PDF file to get started.")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
import streamlit as st
import os, tempfile, asyncio

st.set_page_config(page_title="Legal Doc + Web Assistant")
st.title("⚖️ Legal Assistant – Ask legal questions from documents or websites")

load_dotenv()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not set in .env file.")
    st.stop()

# Choose between PDF or URL
input_type = st.radio("Choose input source:", ("Upload PDF", "Enter Website URL"))

docs = []

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        st.write(f"Uploaded file: {uploaded_file.name}, size: {uploaded_file.size} bytes")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()
        

elif input_type == "Enter Website URL":
    user_url = st.text_input("Enter a website URL to crawl:")
    if user_url:
        async def crawl_website(url):
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                return result.markdown
        markdown_content = asyncio.run(crawl_website(user_url))
        from langchain_core.documents import Document
        docs = [Document(page_content=markdown_content)]

# Process the docs if available
if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_store")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    query = st.text_input("Ask your question:")

    if query:
        legal_prompt = f"You are a legal assistant. Answer the following legal question based only on the provided content. Be precise and avoid assumptions.\n\nQuestion: {query}"
        response = qa_chain.invoke({"query": legal_prompt})
        st.write("📌 **Answer:**", response["result"])
        st.markdown("🔍 **Source Excerpts:**")
        for doc in response["source_documents"]:
            st.markdown(f"```{doc.page_content[:400]}...```")
    else:
        st.write("Please enter a question to continue.")
else:
    st.info("Upload a PDF or enter a website to get started.")
