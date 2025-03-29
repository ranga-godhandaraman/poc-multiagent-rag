import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.llms import Cohere
from langchain.schema import Document
import os
from pathlib import Path
import time
from tqdm import tqdm

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient embedding model
DB_DIR = "./vector_db"
TEMP_UPLOAD_DIR = "temp_uploads"
FEEDBACK_FILE = "feedback_log.jsonl"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
BATCH_SIZE = 64  # Optimized batch size for faster processing
CHUNK_SIZE = 800  # Smaller chunks for faster processing
CHUNK_OVERLAP = 100  # Reduced overlap for efficiency

# Setup Vector DB with caching and optimized settings
def initialize_vector_db():
    # Create the directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)

    # Initialize embeddings with caching for better performance
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder="./embeddings_cache",  # Cache embeddings locally
        encode_kwargs={'batch_size': BATCH_SIZE}  # Optimize batch processing
    )

    # Initialize Chroma with optimized settings
    try:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding,
            collection_name="documents"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector database: {str(e)}")
        # Create a new collection if there was an error
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding,
            collection_name="documents"
        )

# Enhanced Document Ingestion with better multi-page support
def ingest_documents(files):
    """
    Process and ingest multiple document files (PDF or TXT)

    Args:
        files: List of uploaded file objects

    Returns:
        List of processed document chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = []
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    # Create a text splitter with optimized settings for faster processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Smaller chunks for faster processing
        chunk_overlap=CHUNK_OVERLAP,  # Reduced overlap for efficiency
        length_function=len,
    )

    for file in files:
        # Create a safe filename to avoid any path issues
        safe_filename = "".join([c for c in file.name if c.isalnum() or c in "._- "]).rstrip()
        temp_path = os.path.join(TEMP_UPLOAD_DIR, safe_filename)

        # Save the file temporarily
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        try:
            if file.name.lower().endswith('.pdf'):
                # Use PyMuPDFLoader for PDFs - better multi-page support
                loader = PyMuPDFLoader(temp_path)
                # Load the document
                loaded_docs = loader.load()

                # Split the document into chunks if it's large
                if len(loaded_docs) > 0:
                    # Process each page
                    for i, doc in enumerate(loaded_docs):
                        # Add detailed metadata
                        doc.metadata['source'] = file.name
                        doc.metadata['page'] = i + 1
                        doc.metadata['file_type'] = 'pdf'

                        # For very long pages, split them further
                        if len(doc.page_content) > 1500:
                            split_docs = text_splitter.split_documents([doc])
                            # Preserve page number in split chunks
                            for split_doc in split_docs:
                                split_doc.metadata['page'] = f"{i + 1} (section)"
                            docs.extend(split_docs)
                        else:
                            docs.append(doc)
                else:
                    # Handle empty PDFs
                    empty_doc = Document(
                        page_content="[This PDF appears to be empty or could not be processed]",
                        metadata={'source': file.name, 'page': 'N/A', 'file_type': 'pdf'}
                    )
                    docs.append(empty_doc)

            else:
                # Use TextLoader for text files
                loader = TextLoader(temp_path)
                loaded_docs = loader.load()

                # Process text files - split into manageable chunks
                if loaded_docs:
                    split_docs = text_splitter.split_documents(loaded_docs)

                    # Add metadata to each chunk
                    for i, doc in enumerate(split_docs):
                        doc.metadata['source'] = file.name
                        doc.metadata['chunk'] = i + 1
                        doc.metadata['file_type'] = 'txt'
                        doc.metadata['total_chunks'] = len(split_docs)

                    docs.extend(split_docs)
                else:
                    # Handle empty text files
                    empty_doc = Document(
                        page_content="[This text file appears to be empty]",
                        metadata={'source': file.name, 'chunk': 'N/A', 'file_type': 'txt'}
                    )
                    docs.append(empty_doc)

        except Exception as e:
            # Add an error document so the user knows something went wrong
            error_doc = Document(
                page_content=f"[Error processing file: {str(e)}]",
                metadata={'source': file.name, 'error': str(e)}
            )
            docs.append(error_doc)

        finally:
            # Always clean up the temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return docs

# Optimized Embedding and Indexing
def index_documents(vector_store, docs):
    """
    Efficiently index documents with optimized batch processing

    Args:
        vector_store: The vector database
        docs: List of documents to index
    """
    # Get all existing document IDs
    if hasattr(vector_store, "_collection") and vector_store._collection is not None:
        # Get all IDs if there are any documents
        ids = vector_store._collection.get()["ids"]
        if ids:
            vector_store.delete(ids=ids)

    # Add new documents in optimized batches
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Indexing documents"):
        batch = docs[i:i + BATCH_SIZE]
        vector_store.add_documents(batch)

        # Persist after each batch to avoid memory issues with large document sets
        if i % (BATCH_SIZE * 4) == 0 and i > 0:
            vector_store.persist()

    # Final persistence
    vector_store.persist()

# Enhanced prompt template with better context handling
def create_prompt_template():
    template = """
    You are a helpful assistant analyzing the following documents:
    {context}
    
    Based on this information, please answer the following question:
    {question}
    
    Guidelines:
    1. Only use information from the provided documents
    2. If the question cannot be answered from the documents, say "I don't have enough information to answer this question"
    3. Be concise but thorough in your response
    4. If relevant, include page numbers or document names in your response
    5. Do not use first person pronouns ("I") in your response
    
    Answer:
    """
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

# Function to save feedback to a file
def save_feedback(feedback_data):
    """
    Save feedback data to a JSONL file

    Args:
        feedback_data (dict): Dictionary containing feedback information
    """
    try:
        with open(FEEDBACK_FILE, 'a') as f:
            import json
            f.write(json.dumps(feedback_data) + '\n')
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False

# Enhanced RAG-based QA with Cohere
def rag_qa(vector_store, query):
    if not COHERE_API_KEY:
        st.error("Cohere API Key is missing. Set COHERE_API_KEY in your environment.")
        return "", []

    # Check if vector store has documents
    if hasattr(vector_store, "_collection") and vector_store._collection is not None:
        # Try to get document count
        try:
            count = vector_store._collection.count()
            if count == 0:
                return "No documents found in the database. Please upload documents first.", []
        except Exception:
            # If count method fails, continue anyway
            pass

    # Use a faster retriever with optimized settings
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Reduced from 5 for faster response
    )

    # Create and use the enhanced prompt template
    prompt_template = create_prompt_template()

    llm_chain = LLMChain(
        llm=Cohere(
            model="command-xlarge",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY,
            # No token limit for unrestricted output
        ),
        prompt=prompt_template
    )

    combine_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_chain,
        return_source_documents=True
    )

    # Process the query without additional spinner
    try:
        response = qa_chain({"query": query})
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return "Failed to process your query. The database might be empty or there was an error.", []

    # Robust handling of missing source_documents
    sources = []
    if isinstance(response, dict) and 'source_documents' in response:
        sources = [f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" for doc in response['source_documents']]
    else:
        sources.append("Source information not available.")

    if 'result' in response:
        return response['result'], sources
    else:
        return "No response available", sources

# Function to load feedback from file
def load_feedback():
    """
    Load feedback data from the JSONL file

    Returns:
        list: List of feedback entries
    """
    feedback_entries = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                import json
                for line in f:
                    if line.strip():
                        feedback_entries.append(json.loads(line))
            return feedback_entries
        except Exception as e:
            print(f"Error loading feedback: {str(e)}")
    return []

# Streamlit UI
st.set_page_config(page_title="AI-Powered Document Assistant", page_icon="🤖", layout="wide")
st.title("AI-Powered Document Assistant")

# Initialize vector database at startup
vector_db = initialize_vector_db()

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []  # List to store all Q&A pairs

# Document upload section with enhanced support for multiple files
st.markdown("### Upload Documents")
st.markdown("Upload up to 5 PDF or TXT files (multi-page documents supported)")

# Create a container for the uploader with custom styling
upload_container = st.container()
with upload_container:
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload up to 5 PDF or TXT files. Multi-page documents are fully supported."
    )

# Display information about uploaded files
if uploaded_files:
    # Check if we have too many files
    if len(uploaded_files) > 5:
        st.warning(f"You've uploaded {len(uploaded_files)} files. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    # Show file information
    st.markdown("### Uploaded Files:")
    for i, file in enumerate(uploaded_files):
        file_size = round(file.size / 1024, 1)  # Convert to KB

        # Get file extension and show appropriate icon
        if file.name.endswith('.pdf'):
            icon = "📄"
        else:
            icon = "📝"

        st.markdown(f"{i+1}. {icon} **{file.name}** ({file_size} KB)")

    # Process button with progress tracking
    if st.button("Process Documents", key="process_docs"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # First update - starting
        status_text.text("Starting document processing...")
        progress_bar.progress(10)

        try:
            # Second update - ingesting
            status_text.text("Ingesting documents...")
            documents = ingest_documents(uploaded_files)
            progress_bar.progress(50)

            # Third update - indexing
            status_text.text("Indexing documents for search...")
            index_documents(vector_db, documents)
            progress_bar.progress(90)

            # Final update
            status_text.text("Finalizing...")
            progress_bar.progress(100)

            # Success message with document count
            st.success(f"Successfully processed {len(uploaded_files)} documents with a total of {len(documents)} pages/sections!")

            # Clear status elements
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Add CSS for all buttons
st.markdown(
    """
    <style>
    /* Ask button */
    div.stButton > button[data-testid="ask_button"] {
        margin-top: 24px;
        height: 38px;
        padding: 0px 10px;
        width: auto;
        min-width: 80px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 400;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    div.stButton > button[data-testid="ask_button"]:hover {
        background-color: #45a049;
    }

    /* Clear History button */
    div.stButton > button[data-testid="clear_history"] {
        background-color: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
    }
    div.stButton > button[data-testid="clear_history"]:hover {
        background-color: #d32f2f;
    }

    /* Helpful button */
    div.stButton > button[data-testid*="helpful"] {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
        width: auto;
        min-width: 80px;
    }
    div.stButton > button[data-testid*="helpful"]:hover {
        background-color: #45a049;
    }

    /* Not Helpful button */
    div.stButton > button[data-testid*="not_helpful"] {
        background-color: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
        width: auto;
        min-width: 80px;
    }
    div.stButton > button[data-testid*="not_helpful"]:hover {
        background-color: #d32f2f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Check if vector database has documents
if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    # Initialize session state
    if 'responses' not in st.session_state:
        st.session_state.responses = []  # List to store all Q&A pairs

    # Display a welcome message if no responses yet
    if not st.session_state.responses:
        st.markdown("### Welcome! Ask a question about your documents below.")

    # Display all previous responses
    for i, item in enumerate(st.session_state.responses):
        # Display user query in a chat-like format
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {item['query']}</div>", unsafe_allow_html=True)

        # Display assistant response in a chat-like format
        st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Assistant:</strong> {item['response']}</div>", unsafe_allow_html=True)

        # Display sources in a collapsible section
        with st.expander("View Sources"):
            st.write("\n".join(item['sources']))

        # Feedback buttons section
        if item['feedback'] is None:
            col1, col2, col3 = st.columns([1, 1, 4])

            # Use unique keys for each button based on timestamp to avoid conflicts
            key_suffix = item['timestamp'].replace(" ", "_").replace(":", "")

            if col1.button("👍 Helpful", key=f"helpful_{key_suffix}"):
                item['feedback'] = 'positive'

                # Log positive feedback
                feedback_log = {
                    'query': item['query'],
                    'response': item['response'],
                    'sources': item['sources'],
                    'feedback': 'positive',
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                save_feedback(feedback_log)
                st.success("Thanks for your feedback!")
                st.rerun()

            if col2.button("👎 Not Helpful", key=f"not_helpful_{key_suffix}"):
                item['feedback'] = 'negative'

                # Log negative feedback
                feedback_log = {
                    'query': item['query'],
                    'response': item['response'],
                    'sources': item['sources'],
                    'feedback': 'negative',
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                save_feedback(feedback_log)
                st.success("Thanks for your feedback!")
                st.rerun()
        else:
            # Show that feedback was provided in a more subtle way
            st.markdown(f"<div style='color: #666; font-size: 12px; margin-top: 5px;'>Feedback: {'👍 Helpful' if item['feedback'] == 'positive' else '👎 Not Helpful'}</div>", unsafe_allow_html=True)

        # Add a divider between Q&A pairs
        st.markdown("---")

    # Add a button to clear history
    if st.session_state.responses:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Clear History", key="clear_history"):
                st.session_state.responses = []
                st.rerun()
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    # Always show a single input box at the bottom
    with st.form(key='query_form', clear_on_submit=True):
        query = st.text_input("Ask a question about your documents")
        submit_button = st.form_submit_button("Ask")

        if submit_button and query:
            try:
                with st.spinner('Processing your query...'):
                    # Process the query
                    response, sources = rag_qa(vector_db, query)

                    # Add to responses
                    st.session_state.responses.append({
                        'query': query,
                        'response': response,
                        'sources': sources,
                        'feedback': None,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.rerun()  # Rerun to show the new response
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

else:
    st.info("Please upload documents to start asking questions.")
