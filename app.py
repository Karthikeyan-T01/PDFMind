import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Fixed embedding model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv   
from langchain_community.embeddings import HuggingFaceEmbeddings

import time

# Load .env
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if API is loaded
if not groq_api_key:
    st.error("GROQ_API_KEY is missing! Please check your .env file.")
    st.stop()

# Initialized the Groq LLM model.
llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context .
    Please provide the most accurate response based on the question.
    <context>
    {context} 
    <context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()  # Hugging Face embeddings
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Load PDF
        
        # Load documents
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")
        
        # Check if documents are loaded
        if not st.session_state.docs:
            st.error("No documents found. Please check the 'research_papers' directory.")
            return  # Stop execution if no documents
        
        # Split documents into smaller chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        if not st.session_state.final_documents:
            st.error("No document chunks created. Check text splitting settings.")
            return
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector database created successfully!")

# Streamlit UI
st.title("RAG Document Q&A With Groq API")

# User input
user_prompt = st.text_input("Enter your query from the PDF")

# Button to trigger document embedding
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is Created")

# Processing user query 
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please create the document embeddings by clicking ->Document Embedding .")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start} seconds")

        # Display answer
        st.write(response['answer'])

        # Show document similarity search results
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
