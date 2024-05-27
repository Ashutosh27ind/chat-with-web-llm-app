---

# Frog Bot 🐸🤖🔗
## Chat with any complex website
Frog Bot is a conversational AI application built using Streamlit and LangChain technologies. It allows users to interact with an AI agent trained on website content to provide informative responses to user queries.
---

## Folder Structure

```bash
.
|__docs                      # Vector DB directory
|   |__chroma                
|__logs                      # App Logs directory
|   |__app.log               # App log file
|__notebooks
|__resources                 # Project resource file 
|__.env                      # Environment variable configuration
|__.gitignore                # Gitignore file
|__app.py                    # Main application file
|__README.md                 # Project README file
|__LICENSE.md                # License file
|__requirements.txt          # Python dependencies file

```

## Prerequisites
1. Sign up for an [Apify](https://console.apify.com/sign-up) 
2. Create an [OpenAI](https://openai.com/) account and obtain your personal API keys.
3. Install Python 3.10 or higher.

## Setup Instructions

To set up and run this project, follow these steps:

1. **Clone the Repository:**
   ```
   git clone https://github.com/Ashutosh27ind/chat-with-web-llm-app.git
   cd chat-with-web-llm-app
   ```

2. **Install the required dependencies** with `pip`:
   ```
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**: Modify the `.env` file and replace the variables. Here's an explanation of the variables in the env file:
  
`OPENAI_API_KEY`: Your OpenAI API key. You can obtain it from your OpenAI account dashboard.  
`APIFY_API_TOKEN`: Your Apify API token. You can obtain it from [Apify settings](https://console.apify.com/account/integrations).

4. **Run the Streamlit chat application** : 
   1. First, it scrapes the website's data using Apify's Website content crawler. This covers scraping, data cleaning, embedding, splitting, and persisting the data in a vector database.  
   2. Then, it runs the Streamlit app, which should default to http://localhost:8501 and allow you to chat with the website if you are running locally:   
   ```
   cd src
   streamlit run app.py
   ```

## Usage

1. Enter the URL of the website you want to inquire about in the sidebar.
2. Type your message/query in the input box and press Enter.
3. View the AI agent's response in the chat interface.
4. Optionally, you can clear the chat history by clicking the "Clear message history" button in the sidebar.

## Code Explanation:

This code creates an interactive Streamlit application for users to ask questions about a specified website. It leverages a combination of a language model and a vector store retriever to provide responses. The application is designed to maintain a chat history and allows users to clear it as needed.

1. **Environment Configuration**:  
   - The code starts by importing necessary libraries and setting up logging configurations.  
   - It loads environment variables from a `.env` file using `load_dotenv()`.  
   - API keys for OpenAI and Apify are retrieved from environment variables, and the code ensures they are set, raising an error if they are missing.  
          
2. **Logging Setup:**     
   - A logging directory (`logs`) is created if it does not exist, and logging is configured to write to both a log file (`app.log`) and the console.  
   - The logging configuration includes setting the log level, format, and date format.  
   
3. **Text Cleaning Function**:    
   - The `clean_text()` function removes redundant whitespaces, newlines, and email addresses from the text. It ensures the text is clean and standardized for further processing.   
      
4. **Vector Store Retrieval:**      
   - The `get_vectorstore_from_url()` function uses Apify to scrape website content and create a vector store:     
      - **Apify Crawler**:  
         - An Apify actor (`apify/website-content-crawler`) is initialized to crawl the website.     
         - The crawler starts from the given URL and retrieves the text content of the web pages.    
         - The retrieved content is saved into `Document` objects, each containing the text and the source URL.  
      - **Document Processing**:    
         - The text content from the documents is split into chunks using `RecursiveCharacterTextSplitter`.  
         - Each chunk is cleaned using the `clean_text()` function.  
      - **Vector Store Creation**:  
        - The cleaned chunks are converted into `Document` objects and stored in a `Chroma` vector store.  
        - The vector store is then persisted for future use.  
            
5. **Context Retriever Chain:**
   - The `get_context_retriever_chain()` function creates a context retriever chain:  
     - It initializes a language model (`ChatOpenAI`).  
     - A retriever is created from the vector store.  
     - A prompt template is defined for generating search queries based on the conversation history and user input.  
     - The context retriever chain is created using `create_history_aware_retriever()`.    
       
6. **Conversational RAG Chain:**
   - The `get_conversational_rag_chain()` function creates a conversational Retriever-And-Generator (RAG) chain:      
     - It initializes a language model (`ChatOpenAI`).  
     - A prompt template is defined for the chat conversation, including context and chat history.  
     - A document chain is created using `create_stuff_documents_chain()`.  
     - The conversational RAG chain is created by combining the retriever chain and the document chain using `create_retrieval_chain()`.  
       
7. **Generating Responses:**  
   - The `get_response()` function generates a response based on user input and chat history:  
     - It retrieves the context retriever chain and the conversational RAG chain.  
     - The user input and chat history are passed to the RAG chain to generate a response.   
     - The response is returned to the user.  
     
8. **Streamlit App Setup:**
   - The Streamlit application is configured with a title, sidebar, and main content area:
     - The sidebar allows users to input the website URL and clear the chat history.
     - If a URL is provided, the vector store is retrieved and stored in the session state.
     - Users can type their messages, which are processed to generate responses.
     - The chat history is displayed, showing both user queries and AI responses.  
        
9. **Styling the App:**
   - Custom CSS is applied to style the Streamlit app, including the background, sidebar, buttons, and text input fields for a professional and elegant appearance.
   
## Contributing

Contributions to Frog Bot are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Don't forget to star this repo if you find it useful!*

---
# chat-with-web-llm-app
