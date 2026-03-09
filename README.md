# 📚 RAG Assistant (Streamlit + LangChain)

An intelligent **Retrieval-Augmented Generation (RAG) assistant** built with **Streamlit**, **LangChain**, **OpenAI**, and **FAISS**.

The application allows users to upload documents and chat with them using AI.

Supported files:

* PDF
* CSV
* DOCX
* XLSX
* TXT
* MD

The system extracts text from files, splits them into chunks, generates embeddings, and stores them in a **FAISS vector database**.
When a user asks a question, the system retrieves the most relevant chunks and sends them to the LLM to generate an accurate answer.

---

#  Features

*  Upload multiple files
*  Semantic search using embeddings
*  AI-powered question answering
*  Support for structured data (CSV / Excel)
*  Source tracking (page / row / sheet)
*  Fast FAISS vector search
*  Built with Streamlit UI

---

# Architecture

The application follows the **RAG pipeline**:

1. Upload documents
2. Extract text from files
3. Split text into chunks
4. Generate embeddings using OpenAI
5. Store embeddings in FAISS vector store
6. Retrieve relevant chunks when the user asks a question
7. Send retrieved context to the LLM
8. Generate the final answer

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Setup OpenAI API Key

Set your OpenAI API key as an environment variable.

Linux / Mac:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows:

```bash
set OPENAI_API_KEY=your_api_key_here
```

Or add it to **Streamlit secrets**.

---

#  Run the Application

```bash
streamlit run app.py
```

The app will start at:

```
http://localhost:8501
```

---

#  How It Works

* Documents are converted into **LangChain Documents**
* Text is split using **RecursiveCharacterTextSplitter**
* Embeddings are created with **OpenAI Embeddings**
* Stored inside **FAISS vector store**
* Queries retrieve the **top 4 relevant chunks**
* Context is sent to **GPT model**
* The assistant responds based on retrieved data

---

#  Project Structure

```
rag-assistant/
│
├── main.py
├── requirements.txt
├── README.md
└── assets/
```

---

#  Supported File Processing

| File Type | Processing Method    |
| --------- | -------------------- |
| PDF       | Page extraction      |
| DOCX      | Paragraphs + tables  |
| CSV       | Row by row           |
| XLSX      | Sheet + row indexing |
| TXT / MD  | Full text            |

---

#  Future Improvements

* Add **document preview**
* Support **images and OCR**
* Add **multi-language UI**
* Use **local embeddings**
* Deploy on **Streamlit Cloud / Docker**

---

#  License

MIT License

---

#  Author

Developed as a **RAG assistant project using LangChain and Streamlit**.
