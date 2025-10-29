# RAG Application 🚀

A comprehensive Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content using advanced embedding models and Large Language Models (LLMs).

## Features

- **Multi-format Document Support**: Upload PDF, TXT, or DOCX files
- **Query Expansion**: Automatically expands user queries into multiple specific sub-queries for better retrieval
- **Vector Similarity Search**: Uses FAISS for efficient semantic search
- **LLM-powered Synthesis**: Generates comprehensive answers using retrieved context
- **Web Interface**: Clean Streamlit-based user interface
- **Fallback Data**: Uses built-in data.txt if no file is uploaded

## Architecture

The application follows a typical RAG pipeline:

1. **Document Processing**: Splits documents into chunks using RecursiveCharacterTextSplitter
2. **Embedding Generation**: Converts text chunks to vectors using NVIDIA's embedding model
3. **Vector Storage**: Stores embeddings in FAISS index for fast similarity search
4. **Query Processing**: Expands user queries and retrieves relevant chunks
5. **Answer Generation**: Uses LLM to synthesize final answers from retrieved context

## Prerequisites

- Python 3.8+
- Access to embedding API endpoint (`http://10.25.37.28:8000`)
- Access to LLM API endpoint (`http://10.25.37.1:32003`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Streamlit Web Application

Run the web interface:
```bash
streamlit run app.py
```

1. Open your browser and navigate to the provided URL (typically `http://localhost:8501`)
2. Upload a document (PDF, TXT, or DOCX) or leave blank to use the default data
3. Enter your question in the text input field
4. Click "Generate Answer" to get your response

### Option 2: Command Line Interface

For offline processing and querying:

1. **Prepare embeddings** (run once):
```bash
python embed_chunks.py
```

2. **Ask questions**:
```bash
python rag_ans.py
```

## File Structure

```
rag-application/
├── app.py                 # Main Streamlit application
├── rag_ans.py            # Command-line RAG interface
├── embed_chunks.py       # Embedding generation script
├── load_chunk.py         # Document loading and chunking utility
├── data.txt              # Default RAG knowledge base
├── requirements.txt      # Python dependencies
├── faiss_index.bin       # Generated FAISS index (after running embed_chunks.py)
└── text_chunks.pkl       # Serialized text chunks (after running embed_chunks.py)
```

## API Dependencies

### Embedding API
- **Endpoint**: `http://10.25.37.28:8000/v1/embeddings`
- **Model**: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- **Purpose**: Converts text to numerical vectors

### LLM API
- **Endpoint**: `http://10.25.37.1:32003/v1/chat/completions`
- **Model**: `meta/llama-3.1-70b-instruct`
- **Purpose**: Query expansion and answer synthesis

## Configuration

### Chunking Parameters
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Splitter**: RecursiveCharacterTextSplitter

### Retrieval Parameters
- **Top-K per Query**: 2 chunks
- **Query Expansion**: 3-5 expanded queries per user question
- **Vector Search**: L2 distance using FAISS

### LLM Parameters
- **Max Tokens**: 4096 (synthesis), 200 (expansion)
- **Temperature**: 0.3 (synthesis), 0.2 (expansion)
- **Top-P**: 0.9

## Supported File Formats

| Format | Extension | Library Used |
|--------|-----------|--------------|
| PDF | `.pdf` | PyPDF2 |
| Text | `.txt` | Built-in Python |
| Word Document | `.docx` | python-docx |

## Error Handling

The application includes comprehensive error handling for:
- API connection failures
- File reading errors  
- Embedding generation failures
- Invalid file formats
- Network timeouts

## Development

### Adding New File Formats

To support additional file formats, extend the file reading functions in `app.py`:

```python
def read_new_format(file):
    # Your implementation here
    return extracted_text
```

### Customizing Chunk Size

Modify the splitter parameters in the respective files:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=YOUR_SIZE,
    chunk_overlap=YOUR_OVERLAP
)
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Verify that the embedding and LLM API endpoints are accessible
2. **File Upload Failures**: Ensure uploaded files aren't corrupted and are in supported formats
3. **Memory Issues**: For large documents, consider reducing chunk size or processing in batches
4. **Slow Performance**: Check network connectivity to API endpoints

### Debug Mode

For additional debugging information, monitor the Streamlit console output and API response logs.

## Acknowledgments

- NVIDIA for the embedding models
- Meta for the Llama language models
- Streamlit for the web framework
- FAISS for efficient vector search
- LangChain for document processing utilities