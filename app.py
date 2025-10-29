import streamlit as st
import requests
import numpy as np
import faiss
from PyPDF2 import PdfReader
from docx import Document
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
import json
import os
import re
try:
    from dotenv import load_dotenv  
    load_dotenv()
except Exception:
    pass
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def read_txt(file):
    return StringIO(file.getvalue().decode("utf-8")).read()

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_data_txt():
    return """RAG stands for Retrieval-Augmented Generation.
It is a technique used in modern question answering systems where documents are first split into chunks.
Each chunk is converted inext into a large language model (LLM) to generate a response.
This architecture allows LLMs to answer questions using private, domain-specific, or updated information.to a numerical vector using an embedding model.
These vectors are stored in a vector database such as FAISS or Chroma.
When a user asks a question, the system retrieves the most relevant chunks from the database.
These chunks are passed as cont"""


def embed_chunks(chunks):
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    payload = {"input": chunks, "model": EMBED_MODEL, "input_type": "passage"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        st.error(f"Embedding API error: {e}. Body: {body}")
        return None

def get_query_embedding(question):
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    payload = {"input": [question], "model": EMBED_MODEL, "input_type": "query"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        st.error(f"Embedding API error: {e}. Body: {body}")
        return None


def expand_query_with_llm(question):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "accept": "application/json", 
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    
    expansion_prompt = f"""You are an expert at converting a user's question into effective search queries for a vector database.
Based on the user's question, generate 3 to 5 specific, diverse questions that would help retrieve the most relevant and comprehensive information.
Return the questions as a numbered list. Do not add any other commentary.

User Question: "{question}"
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": expansion_prompt}],
        "max_tokens": 200, "temperature": 0.2
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        expanded_queries = [line.split('. ', 1)[1] for line in content.strip().split('\n') if '. ' in line]
    
        if question not in expanded_queries:
            expanded_queries.insert(0, question)
            
        return expanded_queries
    except Exception as e:
        st.warning(f"Could not expand query, using original question. Error: {e}")
        return [question] 


def ask_llm(context, question):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "accept": "application/json", 
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    
    synthesis_prompt = f"""You are a helpful AI assistant. Answer the user's question based *only* on the following context.
If the context does not contain the answer, state that you cannot answer with the information provided.
Be comprehensive and synthesize information from all parts of the context.

Context:
{context}

Question: {question}
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": synthesis_prompt}],
        "max_tokens": 4096, "temperature": 0.3, "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        st.error(f"LLM API error: {e}. Body: {body}")
        return None


def ask_llm_open(question):
    """Ask the LLM without any context constraint (general knowledge fallback)."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ],
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        st.error(f"Open LLM API error: {e}. Body: {body}")
        return None


def _looks_like_refusal(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    patterns = [
        "cannot answer",
        "can't answer",
        "not contain the answer",
        "not in the context",
        "information provided",
        "insufficient information",
        "do not have enough information",
    ]
    return any(p in t for p in patterns)

st.title("RAG")

uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX) or leave blank to use data.txt", type=["pdf", "txt", "docx"])
question = st.text_input("Ask a question about the document:")
use_doc_only = st.checkbox("Use only the uploaded document (no external knowledge)", value=bool(uploaded_file))

if st.button("Generate Answer") and question.strip():
    with st.spinner("Reading and processing file..."):
        file_text = ""
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                file_text = read_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                file_text = read_txt(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_text = read_docx(uploaded_file)
        else:
            file_text = read_data_txt()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        raw_chunks = splitter.split_documents([LCDocument(page_content=file_text)])
        chunk_texts = [chunk.page_content for chunk in raw_chunks]

    with st.spinner("Embedding document chunks..."):
        chunk_embeddings = embed_chunks(chunk_texts)
        if not chunk_embeddings:
            if use_doc_only:
                st.error("Embedding failed for the document; cannot answer strictly from the document.")
                st.stop()
            else:
                st.warning("Embedding failed; falling back to general knowledge answer.")
                fallback = ask_llm_open(question)
                if fallback:
                    st.success("Answer (no document context):")
                    st.write(fallback)
                else:
                    st.error("Could not generate an answer.")
                st.stop()

        index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
        index.add(np.array(chunk_embeddings).astype("float32"))

    with st.spinner("Expanding query and retrieving relevant chunks..."):
        expanded_queries = expand_query_with_llm(question)
        st.info(f"Expanded Queries: {expanded_queries}")

        retrieved_indices = set()
        for q in expanded_queries:
            query_emb = get_query_embedding(q)
            if query_emb:
                D, I = index.search(np.array([query_emb]).astype("float32"), k=2) 
                for i in I[0]:
                    retrieved_indices.add(i)

        if not retrieved_indices:
            if use_doc_only:
                query_emb = get_query_embedding(question)
                if query_emb is not None:
                    D, I = index.search(np.array([query_emb]).astype("float32"), k=min(5, len(chunk_texts)))
                    retrieved_indices = set([int(i) for i in I[0] if i >= 0])
                if not retrieved_indices:
                    retrieved_indices = set(range(min(5, len(chunk_texts))))
                st.info("Using the closest document chunks only (no external knowledge).")
            else:
                st.info("No relevant chunks found. Answering from general knowledge.")
                fallback = ask_llm_open(question)
                if fallback:
                    st.success("Answer (no document context):")
                    st.write(fallback)
                else:
                    st.error("Could not generate an answer.")
                st.stop()

        top_chunks = [chunk_texts[i] for i in retrieved_indices]
        context = "\n\n---\n\n".join(top_chunks)
    
    with st.spinner("Synthesizing the answer with LLM..."):
        answer = ask_llm(context, question)
        if not answer or _looks_like_refusal(answer):
            if use_doc_only:
                st.warning("The document doesn't appear to contain the answer. Try rephrasing or uploading a document that covers this topic.")
            else:
                st.info("The document didn't contain the answer. Answering from general knowledge.")
                answer = ask_llm_open(question)
        if answer:
            st.success("Answer:")
            st.write(answer)
        else:
            if not use_doc_only:
                st.error("Could not generate an answer.")
            else:
                st.error("Could not answer strictly from the document.")