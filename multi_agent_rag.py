import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.llms import Cohere
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, TypedDict, Annotated
import json
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Constants
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 

# Configure paths for data persistence
DATA_DIR = os.getenv('DATA_DIR', '/app/data') 
DB_DIR = os.path.join(DATA_DIR, "vector_db")
TEMP_UPLOAD_DIR = os.path.join(DATA_DIR, "temp_uploads")
EMBEDDINGS_CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.jsonl")

# API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set")

# Processing parameters
BATCH_SIZE = 32  # Balanced batch size for quality and speed
CHUNK_SIZE = 500  # Smaller chunks for more granular retrieval
CHUNK_OVERLAP = 150  # Increased overlap to maintain context
DEFAULT_RETRIEVE_COUNT = 6  # Retrieve more documents by default

# Define the state for our multi-agent system
class AgentState(TypedDict):
    """State for the multi-agent system."""
    query: str
    documents: List[Document]
    summary: Optional[str]
    answer: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]
    error: Optional[str]

# Setting up Vector DB 
def initialize_vector_db():
    """
    Initialize the vector database with optimized settings for better cross-document retrieval

    Returns:
        Configured Chroma vector store
    """
    os.makedirs(DB_DIR, exist_ok=True)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=EMBEDDINGS_CACHE_DIR,
        encode_kwargs={
            'batch_size': BATCH_SIZE,
            'normalize_embeddings': True 
        },
        model_kwargs={
            'device': 'cpu' 
        }
    )

    # Initializing Chroma 
    try:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding,
            collection_name="documents",
            collection_metadata={"hnsw:space": "cosine"} 
        )
        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector database: {str(e)}")
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding,
            collection_name="documents",
            collection_metadata={"hnsw:space": "cosine"}
        )

def ingest_documents(files):
    """
    Process and ingest multiple document files (PDF or TXT)

    Args:
        files: List of uploaded file objects

    Returns:
        List of processed document chunks
    """
    docs = []
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    for file in files:
        safe_filename = "".join([c for c in file.name if c.isalnum() or c in "._- "]).rstrip()
        temp_path = os.path.join(TEMP_UPLOAD_DIR, safe_filename)

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            if file.name.lower().endswith('.pdf'):
                loader = PyMuPDFLoader(temp_path)
                loaded_docs = loader.load()

                if len(loaded_docs) > 0:
                    for i, doc in enumerate(loaded_docs):
                        doc.metadata['source'] = file.name
                        doc.metadata['page'] = i + 1
                        doc.metadata['file_type'] = 'pdf'
                        split_docs = text_splitter.split_documents([doc])

                        if len(split_docs) > 1:
                            for j, split_doc in enumerate(split_docs):
                                split_doc.metadata['page'] = i + 1
                                split_doc.metadata['section'] = j + 1
                                split_doc.metadata['total_sections'] = len(split_docs)
                                split_doc.metadata['title'] = os.path.splitext(file.name)[0]
                            docs.extend(split_docs)
                        else:
                            doc.metadata['section'] = 1
                            doc.metadata['total_sections'] = 1
                            doc.metadata['title'] = os.path.splitext(file.name)[0]
                            docs.append(doc)
                else:
                    # Handling the empty PDFs
                    empty_doc = Document(
                        page_content="[This PDF appears to be empty or could not be processed]",
                        metadata={'source': file.name, 'page': 'N/A', 'file_type': 'pdf'}
                    )
                    docs.append(empty_doc)

            else:
                # Use TextLoader for text files- Doing the same for text files
                loader = TextLoader(temp_path)
                loaded_docs = loader.load()

                if loaded_docs:
                    split_docs = text_splitter.split_documents(loaded_docs)
                    for i, doc in enumerate(split_docs):
                        doc.metadata['source'] = file.name
                        doc.metadata['chunk'] = i + 1
                        doc.metadata['file_type'] = 'txt'
                        doc.metadata['total_chunks'] = len(split_docs)
                        doc.metadata['title'] = os.path.splitext(file.name)[0]
                        doc.metadata['content_hash'] = hash(doc.page_content) % 10000000
                        position = i / len(split_docs)
                        if position < 0.33:
                            doc.metadata['position'] = 'beginning'
                        elif position < 0.66:
                            doc.metadata['position'] = 'middle'
                        else:
                            doc.metadata['position'] = 'end'

                    docs.extend(split_docs)
                else:
                    empty_doc = Document(
                        page_content="[This text file appears to be empty]",
                        metadata={'source': file.name, 'chunk': 'N/A', 'file_type': 'txt'}
                    )
                    docs.append(empty_doc)

        except Exception as e:
            error_doc = Document(
                page_content=f"[Error processing file: {str(e)}]",
                metadata={'source': file.name, 'error': str(e)}
            )
            docs.append(error_doc)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return docs

# Enhanced Embedding and Indexing for better cross-document retrieval
def index_documents(vector_store, docs):
    """
    Efficiently index documents with optimized settings for improved retrieval

    Args:
        vector_store: The vector database
        docs: List of documents to index
    """
    # Get all existing document IDs
    if hasattr(vector_store, "_collection") and vector_store._collection is not None:
        try:
            ids = vector_store._collection.get()["ids"]
            if ids:
                vector_store.delete(ids=ids)
                st.info(f"Cleared {len(ids)} previous document chunks from the database.")
        except Exception as e:
            st.warning(f"Could not clear previous documents: {str(e)}. Creating a new collection.")
            vector_store = initialize_vector_db()

    # taking unique to remove duplications
    unique_docs = []
    content_hashes = set()

    for doc in docs:
        content_hash = hash(doc.page_content) % 10000000

        if content_hash not in content_hashes:
            content_hashes.add(content_hash)
            doc.metadata['content_hash'] = content_hash
            unique_docs.append(doc)

    st.info(f"Removed {len(docs) - len(unique_docs)} duplicate chunks, indexing {len(unique_docs)} unique chunks.")

    for i, doc in enumerate(unique_docs):
        doc_id = str(uuid.uuid4())
        doc.metadata['doc_id'] = doc_id

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(0, len(unique_docs), BATCH_SIZE):
        progress_percentage = min(1.0, i / len(unique_docs))
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Indexing documents: {i}/{len(unique_docs)} ({int(progress_percentage*100)}%)")

        batch = unique_docs[i:i + BATCH_SIZE]
        ids = [doc.metadata.get('doc_id') for doc in batch]
        vector_store.add_documents(batch, ids=ids)

        if i % (BATCH_SIZE * 2) == 0 and i > 0:
            vector_store.persist()

    progress_bar.progress(1.0)
    progress_text.text(f"Indexing complete: {len(unique_docs)} documents processed")
    vector_store.persist()

    return len(unique_docs)

# Enhanced document retrieval with multi-query expansion and MMR- it means Maximum Marginal Relevance
def retrieve_documents(vector_store, query, k=None):
    """
    Enhanced retrieval of relevant documents across multiple sources

    Args:
        vector_store: The vector database
        query: User query
        k: Number of documents to retrieve (uses DEFAULT_RETRIEVE_COUNT if None)

    Returns:
        List of retrieved documents from multiple sources
    """
    if k is None:
        k = DEFAULT_RETRIEVE_COUNT

    query_variations = generate_query_variations(query)

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": 0.7
        }
    )
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    all_docs = []

    # First, try MMR with the original query
    mmr_docs = mmr_retriever.get_relevant_documents(query)
    all_docs.extend(mmr_docs)

    # Then try with query variations to get more diverse results
    for variation in query_variations:
        # Use similarity search for variations to get more coverage
        variation_docs = similarity_retriever.get_relevant_documents(variation)
        all_docs.extend(variation_docs)

    unique_docs = []
    seen_content = set()

    for doc in all_docs:
        content_hash = hash(doc.page_content) % 10000000

        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)

            if len(unique_docs) >= k * 2:
                break

    # Sort by relevance to original query (approximately by checking if query terms are in the document)
    query_terms = set(query.lower().split())

    def relevance_score(doc):
        content = doc.page_content.lower()
        term_matches = sum(1 for term in query_terms if term in content)
        return term_matches

    # Sorting by relevancy score
    unique_docs.sort(key=relevance_score, reverse=True)
    return unique_docs[:k]


def generate_query_variations(query):
    """
    Generate variations of the query to improve retrieval

    Args:
        query: Original user query

    Returns:
        List of query variations
    """
    variations = []

    variations.append(f"about {query}")
    variations.append(f"information about {query}")
    variations.append(f"explain {query}")

    words = query.split()
    if len(words) > 3:
        # For longer queries, create a keywords-only version
        # Filter out common stop words (simplified)
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word.lower() not in stop_words]
        if keywords:
            variations.append(" ".join(keywords))

    return variations

# Agent 1: Summarizer Agent
def summarizer_agent(state: AgentState) -> AgentState:
    """
    Summarizes the retrieved documents to provide context for the Q&A agent.
    
    Args:
        state: Current state containing documents and query
        
    Returns:
        Updated state with summary
    """
    try:
        documents = state["documents"]
        query = state["query"]
        
        if not documents:
            return {**state, "summary": "No relevant documents found.", "error": "No documents to summarize"}
        
        combined_content = "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}" 
                                      for i, doc in enumerate(documents)])
        prompt_template = """
        You are a document summarizer. Your task is to create a concise summary of the following documents
        that will be used to answer this query: "{query}"
        
        Documents:
        {documents}
        
        Create a comprehensive summary that captures the key information relevant to the query.
        Focus on facts and information that would be helpful for answering the query.
        
        Summary:
        """
        
        prompt = PromptTemplate(
            input_variables=["documents", "query"],
            template=prompt_template
        )
        
        llm = Cohere(
            model="command-xlarge",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        summary = chain.run(documents=combined_content, query=query)
        
        return {**state, "summary": summary}
    except Exception as e:
        return {**state, "error": f"Error in summarizer agent: {str(e)}"}

# Agent 2: Q&A Agent
def qa_agent(state: AgentState) -> AgentState:
    """
    Generates an answer based on the documents and summary.
    
    Args:
        state: Current state containing documents, summary, and query
        
    Returns:
        Updated state with answer
    """
    try:
        documents = state["documents"]
        summary = state["summary"]
        query = state["query"]
        
        if not documents:
            return {**state, "answer": "I don't have enough information to answer this question."}
        
        prompt_template = """
        You are a helpful assistant answering questions based on the provided documents.
        
        QUERY: {query}
        
        DOCUMENT SUMMARY: {summary}
        
        FULL DOCUMENTS:
        {documents}
        
        Based on the provided information, please answer the query thoroughly and accurately.
        Only use information from the provided documents. If the answer cannot be found in the documents,
        state that you don't have enough information to answer the question.
        
        Include specific details and facts from the documents to support your answer.
        
        ANSWER:
        """
        
        prompt = PromptTemplate(
            input_variables=["documents", "summary", "query"],
            template=prompt_template
        )
        
        combined_content = "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}, "
                                       f"Page: {doc.metadata.get('page', 'N/A')}): {doc.page_content}" 
                                      for i, doc in enumerate(documents)])
        
        llm = Cohere(
            model="command-xlarge",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run(documents=combined_content, summary=summary, query=query)
        
        return {**state, "answer": answer}
    except Exception as e:
        return {**state, "error": f"Error in Q&A agent: {str(e)}"}

# Agent 3: Citation Verifier Agent
def citation_verifier_agent(state: AgentState) -> AgentState:
    """
    Verifies the answer against the documents and adds citations.
    
    Args:
        state: Current state containing documents, summary, query, and answer
        
    Returns:
        Updated state with verified answer and citations
    """
    try:
        documents = state["documents"]
        answer = state["answer"]
        query = state["query"]
        
        if not documents or not answer:
            return {**state, "citations": [], "final_response": state.get("answer", "")}
        
        prompt_template = """
        You are a citation verifier. Your task is to verify the following answer against the source documents
        and add proper citations.
        
        QUERY: {query}
        
        ANSWER TO VERIFY: {answer}
        
        SOURCE DOCUMENTS:
        {documents}
        
        Please:
        1. Verify that all claims in the answer are supported by the documents
        2. Add citations to the answer in the format [Source: document_name, Page: page_number]
        3. If any claims are not supported, modify the answer to remove or correct them
        4. Return the verified answer with proper citations
        
        VERIFIED ANSWER WITH CITATIONS:
        """
        
        prompt = PromptTemplate(
            input_variables=["documents", "answer", "query"],
            template=prompt_template
        )

        combined_content = "\n\n".join([f"Document {i+1} - Source: {doc.metadata.get('source', 'Unknown')}, "
                                       f"Page: {doc.metadata.get('page', 'N/A')}: {doc.page_content}" 
                                      for i, doc in enumerate(documents)])
        
        llm = Cohere(
            model="command-xlarge",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        verified_answer = chain.run(documents=combined_content, answer=answer, query=query)
        
        citations = []
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            if source in verified_answer or str(page) in verified_answer:
                citations.append({
                    'source': source,
                    'page': page,
                    'content_preview': doc.page_content[:100] + "..."
                })
        
        return {
            **state, 
            "citations": citations,
            "final_response": verified_answer
        }
    except Exception as e:
        return {**state, "error": f"Error in citation verifier agent: {str(e)}", "final_response": state.get("answer", "")}

# Define the multi-agent workflow
def create_agent_workflow():
    """
    Creates a LangGraph workflow with three agents:
    1. Summarizer Agent
    2. Q&A Agent
    3. Citation Verifier Agent
    
    Returns:
        A compiled workflow that can be executed
    """
    # Creating a new graph
    workflow = StateGraph(AgentState)
    
    # Adding nodes for each agent
    workflow.add_node("summarizer", summarizer_agent)
    workflow.add_node("qa", qa_agent)
    workflow.add_node("citation_verifier", citation_verifier_agent)
    
    # Defining the edges between nodes
    workflow.add_edge("summarizer", "qa")
    workflow.add_edge("qa", "citation_verifier")
    workflow.add_edge("citation_verifier", END)
    
    # Setting up the  the entry point
    workflow.set_entry_point("summarizer")
    
    # Compile the workflow
    return workflow.compile()

# Enhanced query processing with improved retrieval and agent orchestration
def process_query_with_agents(vector_store, query):
    """
    Process a user query using the multi-agent system with enhanced retrieval

    Args:
        vector_store: The vector database
        query: User query

    Returns:
        Final response and citations with source information
    """
    try:

        start_time = time.time()
        documents = retrieve_documents(vector_store, query)
        retrieval_time = time.time() - start_time

        if not documents:
            return "No relevant documents found to answer your question. Please try rephrasing your query or upload more documents.", []

        doc_sources = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source in doc_sources:
                doc_sources[source] += 1
            else:
                doc_sources[source] = 1

        source_info = ", ".join([f"{source} ({count} chunks)" for source, count in doc_sources.items()])
        st.info(f"Retrieved {len(documents)} relevant chunks from {len(doc_sources)} documents: {source_info}")

        # Initialize the state
        initial_state = AgentState(
            query=query,
            documents=documents,
            summary=None,
            answer=None,
            citations=None,
            final_response=None,
            error=None
        )

        workflow = create_agent_workflow()

        with st.spinner("Processing with multiple agents..."):
            result = workflow.invoke(initial_state)
        total_time = time.time() - start_time

        if result.get("error"):
            return f"An error occurred: {result['error']}", []

        final_response = result.get("final_response", "No response generated.")
        citations = result.get("citations", [])

        formatted_citations = []
        for cite in citations:
            source = cite.get('source', 'Unknown')
            page = cite.get('page', 'N/A')
            preview = cite.get('content_preview', '')

            if isinstance(page, (int, float)):
                page_info = f"Page {page}"
            else:
                page_info = f"{page}"

            formatted_citations.append(f"Source: {source} ({page_info})")

        performance_note = f"\n\n_Query processed in {total_time:.2f} seconds (retrieval: {retrieval_time:.2f}s)_"

        return final_response + performance_note, formatted_citations
    except Exception as e:
        return f"Error processing query: {str(e)}", []

# Streamlit UI
def main():
    try:
        st.set_page_config(page_title="AI-Powered Document Assistant", page_icon="🤖", layout="wide")
        st.title("AI-Powered Document Assistant - A POC to Middleware")
        st.markdown("### Multi-Agent RAG System")
        
        vector_db = initialize_vector_db()
        
        if 'responses' not in st.session_state:
            st.session_state.responses = []
        
        st.markdown("### Upload Documents")
        st.markdown("Upload up to 5 PDF or TXT files (multi-page documents supported)")
        
        upload_container = st.container()
        with upload_container:
            uploaded_files = st.file_uploader(
                "Drag and drop files here",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload up to 5 PDF or TXT files. Multi-page documents are fully supported."
            )
        
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning(f"You've uploaded {len(uploaded_files)} files. Only the first 5 will be processed.")
                uploaded_files = uploaded_files[:5]
            
            st.markdown("### Uploaded Files:")
            for i, file in enumerate(uploaded_files):
                file_size = round(file.size / 1024, 1)
                
                if file.name.endswith('.pdf'):
                    icon = "📄"
                else:
                    icon = "📝"
                    
                st.markdown(f"{i+1}. {icon} **{file.name}** ({file_size} KB)")
            
            if st.button("Process Documents", key="process_docs"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Starting document processing...")
                progress_bar.progress(0.1)  

                try:
                    status_text.text("Ingesting documents...")
                    documents = ingest_documents(uploaded_files)
                    progress_bar.progress(0.5)  # 50%

                    status_text.text("Indexing documents for search...")
                    doc_count = index_documents(vector_db, documents)
                    status_text.text("Finalizing...")
                    progress_bar.progress(1.0)  # 100%

                    st.success(f"Successfully processed {len(uploaded_files)} documents with a total of {doc_count} unique chunks!")

                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
        
        if st.session_state.responses:
            st.markdown("### Previous Questions & Answers")
            
            for i, response in enumerate(st.session_state.responses):
                with st.container():
                    st.markdown(f"**Q: {response['query']}**")
                    st.markdown(f"{response['response']}")
                    
                    if response['sources']:
                        with st.expander("View Sources"):
                            for source in response['sources']:
                                st.markdown(f"- {source}")
                    
                    if i < len(st.session_state.responses) - 1:
                        st.markdown("---")
            
            if st.button("Clear History", key="clear_history"):
                st.session_state.responses = []
                st.rerun()
            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        
        with st.form(key='query_form', clear_on_submit=True):
            query = st.text_input("Ask a question about your documents")
            submit_button = st.form_submit_button("Ask")
            
            if submit_button and query:
                try:
                    with st.spinner('Processing your query with multiple agents...'):
                        response, sources = process_query_with_agents(vector_db, query)
                        
                        st.session_state.responses.append({
                            'query': query,
                            'response': response,
                            'sources': sources,
                            'feedback': None,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.rerun()  
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)
                    st.error(f"Error processing query: {str(e)}")
        

        if not uploaded_files:
            st.info("Please upload documents to start asking questions.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
