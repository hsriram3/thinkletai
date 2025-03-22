# import os
# import pickle
# import streamlit as st

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Set your Gemini API Key
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# st.set_page_config(
#     page_title="Thinklet - Smarter News, Instantly",
#     page_icon="🧠",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )
# st.title("✨ Thinklet: Turn Headlines Into Insights")
# st.sidebar.title("🌐 Feed the Brain — Paste Some Links:")

# # Input 3 URLs
# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     if url.strip():
#         urls.append(url)

# process_url = st.sidebar.button("🔄 Process URLs")
# file_path = "faiss_store_huggingface.pkl"
# placeholder = st.empty()

# from langchain_community.llms import HuggingFaceHub

# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7, "max_length": 512}
# )


# if process_url:
#     if not urls:
#         st.warning("Please enter at least one valid URL.")
#     else:
#         loader = UnstructuredURLLoader(urls=urls)
#         placeholder.info("⏳ Loading data from URLs...")
#         data = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", ".", ","],
#             chunk_size=1000
#         )
#         docs = text_splitter.split_documents(data)
#         placeholder.info("Text split into chunks...")

#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         placeholder.success("Embeddings created and vectorstore built!")

#         with open(file_path, "wb") as f:
#             pickle.dump(vectorstore, f)

#         st.success("🎉 Data processed and saved!")

# # Ask a question
# query = st.text_input("Ask a question about the articles:")
# if query:
#     if not os.path.exists(file_path):
#         st.error("Please process URLs first!")
#     else:
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)

#         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#         result = chain({"question": query}, return_only_outputs=True)

#         st.header("📌 Answer")
#         st.write(result["answer"])

#         sources = result.get("sources", "")
#         if sources:
#             st.subheader("📚 Sources")
#             for src in sources.split("\n"):
#                 st.markdown(f"- {src}")

import os
import pickle
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Page Config
st.set_page_config(
    page_title="🧠 Thinklet - AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9fb;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 8px;
        border: 1px solid #d0d0d0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .answer-box {
        background-color: #edf2fc;
        border-left: 5px solid #6C63FF;
        padding: 1rem;
        border-radius: 8px;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #6C63FF;'>🧠 Thinklet</h1>
        <h4 style='color: #444;'>Your AI-Powered Insight Engine</h4>
        <p style='font-size: 16px;'>Upload articles or links, ask questions, and get intelligent summaries and answers instantly!</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Input
st.sidebar.header("📥 Upload or Input Your Sources")
input_mode = st.sidebar.radio("Choose input method:", ["🔗 URLs", "📄 PDFs"])

urls = []
pdf_docs = []
if input_mode == "🔗 URLs":
    for i in range(3):
        url = st.sidebar.text_input(f"🔗 Enter URL {i+1}")
        if url.strip():
            urls.append(url)
else:
    pdf_uploads = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if pdf_uploads:
        for file in pdf_uploads:
            pdf_docs.append(file)

process_btn = st.sidebar.button("🚀 Process Sources")
file_path = "faiss_store_huggingface.pkl"

# LLM Setup
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Main processing
if process_btn:
    docs = []
    with st.spinner("⏳ Processing your content, please wait..."):
        if input_mode == "🔗 URLs":
            loader = UnstructuredURLLoader(urls=urls)
            raw_data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            for doc in raw_data:
                 doc.metadata["source"] = doc.metadata.get("source", "web")

            docs = text_splitter.split_documents(raw_data)


        elif input_mode == "📄 PDFs":
            for pdf in pdf_docs:
                reader = PdfReader(BytesIO(pdf.read()))
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                docs.append(text)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.create_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    st.success("✅ Content successfully processed and indexed!")
    st.balloons()

# Chat Section
st.markdown("""
    <div style='margin-top: 30px;'>
        <h3 style='color: #333;'>🤖 Ask Anything About the Content</h3>
    </div>
""", unsafe_allow_html=True)

query = st.text_input("💬 What's your question?")

if query:
    if not os.path.exists(file_path):
        st.error("❌ Please upload or input sources first.")
    else:
        with st.spinner("🔍 Analyzing your content with LLM..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

        st.markdown("### ✅ Answer")
        st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)

        sources = result.get("sources", "")
        if sources:
            st.markdown("### 📚 Sources")
            for src in sources.split("\n"):
                st.markdown(f"- {src}")

# Footer
st.markdown("""
    <hr style='margin-top: 2rem;'/>
    <div style='text-align: center; font-size: 14px; color: #888;'>
        Made by Hari | A product of YAARAA
    </div>
""", unsafe_allow_html=True)