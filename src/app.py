import logging
import os
import re
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.utilities import ApifyWrapper

# Load environment variables from a .env file early
load_dotenv()

open_ai_key = os.environ.get('OPENAI_API_KEY')
apify_api_token = os.environ.get('APIFY_API_TOKEN')

# Validate API keys
if not open_ai_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
if not apify_api_token:
    raise ValueError("APIFY_API_TOKEN environment variable is not set.")

# Determine the parent directory of the current working directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Create the log directory if it doesn't exist
log_dir = os.path.join(parent_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    filename=os.path.join(log_dir, 'app.log'),  # Log file path
    level=logging.INFO,  # Set the desired log level
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Enable Console Logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Log environment variables
logging.info("Environment Variables Successfully Imported")

# Create a data directory if it doesn't exist
persist_directory = os.path.join(parent_dir, 'docs/chroma/')
os.makedirs(persist_directory, exist_ok=True)
logging.info(f"PERSIST_DIRECTORY: {persist_directory}")

def clean_text(text):
    """
    Cleans the given text by removing email addresses and redundant whitespaces.
    Args:
        text (str): The text to clean.
    Returns:
        str: The cleaned text.
    """
    try:
        text = re.sub(r'(?!\n)\s+', ' ', text)  # Replace multiple whitespaces (except newlines) with a single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
        text = text.strip()  # Remove leading and trailing whitespace
        text = re.sub(r'\S*@\S*\s?', ' ', str(text))  # Remove all email addresses
        return text
    except Exception as e:
        logging.error(f"Error occurred while cleaning text: {str(e)}")
        return ""

def get_vectorstore_from_url(url):
    """
    Fetches vector store from the provided URL and persists it.
    Args:
        url (str): The URL of the website to fetch data from.
    Returns:
        Chroma: The vector store object.
    """
    try:
        # Instantiate apify actor
        apify = ApifyWrapper()

        # Apify actor crawls the webpages at the urls in the list and saves the text and the url inside a Document object
        loader = apify.call_actor(
            actor_id="apify/website-content-crawler",
            run_input={
                "startUrls": [{"url": url}],
                "proxyConfiguration": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"],
                    "apifyProxyCountry": "IN"
                }
            },
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "", metadata={"source": item["url"]}
            ),
        )

        # Load documents
        documents = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(documents)

        # Clean the text chunks
        cleaned_doc_chunks = [clean_text(doc.page_content) for doc in document_chunks]
        logging.info("Cleaned document chunks successfully.")

        # Create a vectorstore from the chunks
        documents = [Document(page_content=chunk) for chunk in cleaned_doc_chunks]
        vector_store = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=persist_directory)
        logging.info(f"Count of documents in Vector DB is: {vector_store._collection.count()}")
        logging.info("Data persisted in the vector database successfully.")

        # Persist the vector database
        vector_store.persist()
        return vector_store
    except Exception as e:
        logging.error(f"Error occurred while fetching vector store: {str(e)}")
        return None

def get_context_retriever_chain(vector_store):
    """
    Creates a context retriever chain using the provided vector store.
    Args:
        vector_store (Chroma): The vector store object.
    Returns:
        HistoryAwareRetrieverChain: The context retriever chain.
    """
    try:
        llm = ChatOpenAI()
        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

        return create_history_aware_retriever(llm, retriever, prompt)
    except Exception as e:
        logging.error(f"Error occurred while creating context retriever chain: {str(e)}")
        return None

def get_conversational_rag_chain(retriever_chain):
    """
    Creates a conversational RAG (Retriever-And-Generator) chain using the provided retriever chain.
    Args:
        retriever_chain (HistoryAwareRetrieverChain): The history-aware retriever chain.
    Returns:
        RetrievalChain: The conversational RAG chain.
    """
    try:
        llm = ChatOpenAI()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a website agent that is talking with a human. Keep your replies informative and be polite. Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    except Exception as e:
        logging.error(f"Error occurred while creating conversational RAG chain: {str(e)}")
        return None

def get_response(user_input):
    """
    Generates a response based on the user input and the current chat history.
    Args:
        user_input (str): The user input message.
    Returns:
        str: The generated response.
    """
    try:
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        if not retriever_chain:
            return "Error occurred while retrieving data. Please try again later."

        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        if not conversation_rag_chain:
            return "Error occurred while processing conversation. Please try again later."

        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        return response['answer']
    except Exception as e:
        logging.error(f"Error occurred while generating response: {str(e)}")
        return "Error occurred while generating response. Please try again later."

# Streamlit App setup
st.set_page_config(page_title="LLM-AI App", page_icon=":frog:", layout="wide", initial_sidebar_state="expanded")

st.title("Ask the AI Bot")

# Sidebar
with st.sidebar:
    st.header("App Settings")
    website_url = st.text_input("Website URL", key="website_url")
    if st.button('Clear message history'):
        st.session_state.chat_history = []

# Main content
if not website_url:
    st.info("Please enter a website URL")
else:
    # Session state initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am an AI bot. Ask me anything about this website!")]

    if "vector_store" not in st.session_state:
        vector_store = get_vectorstore_from_url(website_url)
        if vector_store:
            st.session_state.vector_store = vector_store
        else:
            st.error("Error occurred while initializing. Please check logs for details.")

    # Chat interaction
    user_query = st.text_input("Type your message here.....", key="user_query")

    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.markdown(f"**AI:** {message.content}")
        elif isinstance(message, HumanMessage):
            st.markdown(f"**You:** {message.content}")

# Apply a nice theme
st.markdown("""
<style>
    .reportview-container {
        background: #f5f5f5;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: #2e3b4e;
        color: #fff;
    }
    .stButton>button {
        background-color: #2e3b4e;
        color: white;
    }
    .stTextInput>div>div>input {
        color: #333;
        background-color: #fff;
    }
</style>
""", unsafe_allow_html=True)
