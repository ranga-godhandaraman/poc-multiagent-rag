# Multi-Agent RAG System

An enhanced Retrieval-Augmented Generation (RAG) system using LangGraph to orchestrate multiple specialized agents for document processing and question answering.

## Docker Deployment

### Prerequisites

- Docker and Docker Compose installed
- Cohere API key
- Python 3.11 or higher

### Quick Start

1. Create a `.env` file with your Cohere API key:
   ```
   COHERE_API_KEY=your_api_key_here
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t ranga2024/poc-multiagent-rag .
   
   docker run -d -p 8501:8501 \
       -v $(pwd)/data:/app/data \
       -e COHERE_API_KEY=your_api_key_here \
       ranga2024/poc-multiagent-rag \
       streamlit run multi_agent_rag.py \
       --server.address=0.0.0.0 \
       --server.port=8501 \
       --server.enableCORS=true \
       --server.enableWebsocketCompression=false \
       --server.maxUploadSize=100 \
       --browser.gatherUsageStats=false
   ```

3. Access the application at:
   - http://localhost:8501
   - http://127.0.0.1:8501
   - http://172.17.0.2:8501 (container IP)

### Troubleshooting

1. **Port Conflict**
   - Stop any process using port 8501
   - Use `docker ps` to find running containers
   - Stop conflicting containers with `docker stop <container_id>`

2. **Connection Issues**
   - Verify Docker container is running: `docker ps`
   - Check container logs: `docker logs <container_id>`
   - Ensure no firewall blocking port 8501

3. **API Key Issues**
   - Confirm COHERE_API_KEY is correctly set
   - Check for typos in the API key
   - Verify API key has necessary permissions

### Running the Docker Image

```bash
# Pull the image from Docker Hub
docker pull ranga2024/poc-multiagent-rag:latest

# Run the Docker image
docker run -p 8501:8501 -e COHERE_API_KEY=your_api_key_here ranga2024/poc-multiagent-rag:latest
```

## Features

- **Multi-Agent Architecture**: Uses three specialized agents (Summarizer, Q&A, Citation Verifier) working together
- **Enhanced Document Processing**: Support for multiple PDF and text files with advanced chunking
- **Optimized Retrieval**: Cross-document retrieval with query expansion and MMR
- **LangGraph Orchestration**: Structured workflow for agent collaboration
- **Interactive UI**: Streamlit-based interface with progress tracking and document management

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

## Performance Considerations

- The application is optimized for quality and speed with:
  - Advanced embedding model (BAAI/bge-small-en-v1.5)
  - Optimized chunking strategy
  - Query expansion techniques
  - Maximum Marginal Relevance for diverse retrieval
