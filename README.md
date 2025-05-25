This project is a Document Question Answering (DocQA) application built with Streamlit, LangChain, FAISS, and Groq’s LLaMA-4 model. It allows users to upload PDF documents, which are automatically processed, split, and converted into semantic embeddings using Hugging Face models. These embeddings are stored in a FAISS vector store to enable fast and accurate similarity search.

When a user asks a question, the app retrieves the most relevant chunks from the PDFs and passes them to Groq’s LLaMA-4 model via LangChain to generate context-aware answers — all through an easy-to-use web interface.

It’s ideal for:

📚 Students who want to ask questions about their course notes

🧠 Researchers who need fast answers from large documents

📄 Anyone working with unstructured PDFs and looking to extract knowledge easily

🌐 You can get the API from " Groq’s ultra-fast API and LLaMA-4 models "
