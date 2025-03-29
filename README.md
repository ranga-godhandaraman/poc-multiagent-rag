# Multi-Agent RAG System

An enhanced Retrieval-Augmented Generation (RAG) system using LangGraph to orchestrate multiple specialized agents for document processing and question answering.

## Features

- **Multi-Agent Architecture**: Uses three specialized agents (Summarizer, Q&A, Citation Verifier) working together
- **Enhanced Document Processing**: Support for multiple PDF and text files with advanced chunking
- **Optimized Retrieval**: Cross-document retrieval with query expansion and MMR
- **LangGraph Orchestration**: Structured workflow for agent collaboration
- **Interactive UI**: Streamlit-based interface with progress tracking and document management

## Requirements

- Python 3.10+
- Cohere API key
- Docker and Docker Compose (for containerized deployment)

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Cohere API key:
   ```
   export COHERE_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```
   streamlit run multi_agent_rag.py
   ```

### Option 2: Docker Deployment

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file with your Cohere API key:
   ```
   COHERE_API_KEY=your_api_key_here
   ```

3. Build and run with Docker Compose:
   ```
   docker-compose up -d
   ```

4. Access the application at http://localhost:8501

## Usage

1. Upload up to 5 PDF or text documents using the file uploader
2. Click "Process Documents" to index the documents
3. Ask questions about the documents in the query box
4. View answers with citations to the source documents
5. Previous questions and answers are saved in the session

## Architecture

### Agents

1. **Summarizer Agent**: Creates a concise summary of retrieved documents
2. **Q&A Agent**: Generates comprehensive answers based on documents and summary
3. **Citation Verifier**: Verifies claims and adds proper citations to the answer

### Workflow

The LangGraph workflow orchestrates the agents in the following sequence:
1. Document retrieval → Summarizer Agent → Q&A Agent → Citation Verifier → Final Response

## Docker Configuration

The application is containerized with Docker for easy deployment:

- **Data Persistence**: All data (vector database, embeddings cache, uploads) is stored in a Docker volume
- **Environment Variables**:
  - `COHERE_API_KEY`: Your Cohere API key
  - `DATA_DIR`: Path for data storage (defaults to `/app/data`)
- **Port**: The application runs on port 8501
- **Health Check**: Built-in health check at `/_stcore/health`

### Customizing Docker Deployment

You can customize the deployment by modifying the environment variables in docker-compose.yml:

```yaml
environment:
  - COHERE_API_KEY=${COHERE_API_KEY}
  - DATA_DIR=/custom/path  # Optional: change data directory
```

## Performance Considerations

- The application is optimized for quality and speed with:
  - Advanced embedding model (BAAI/bge-small-en-v1.5)
  - Optimized chunking strategy
  - Query expansion techniques
  - Maximum Marginal Relevance for diverse retrieval
