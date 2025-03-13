# tap_new

Tap Bonds AI Chatbot Project

Developed a Streamlit-based AI chatbot leveraging GROQ's language models to provide bond market insights, cash flow analysis, and company information to users.

Implemented data loading and validation from Google Drive CSV files, including error handling and data cleaning techniques to ensure data accuracy and reliability.

Designed and implemented a query processing system that analyzes user intent, extracts key information, and retrieves relevant data to provide accurate responses.

Integrated DuckDuckGo Search API to enhance the chatbot's ability to answer complex queries by scraping and parsing relevant information from the web.

Created a user-friendly interface with Streamlit, including interactive elements such as chat history, data loading status indicators, and configuration options for customizing the AI model.

Overview of the Tap Bonds AI Chatbot
The provided code outlines a Streamlit web application designed as an AI chatbot specifically for bond market analysis. The "Tap Bonds AI Chatbot" enables users to retrieve information regarding bonds, cash flows, company insights, and yield calculations. It can also perform web searches for bond market data.

Approach
The chatbot uses a modular design:

Data Management: Loads bond data, cash flow details, and company insights from CSV files stored on Google Drive.

Query Processing: Examines user queries to determine their intent (e.g., bond details, cash flow, company data, yield calculations, or web search).

LLM Integration: Employs GROQ's language models (like Llama3 or Mixtral) to generate natural language responses.

Web Search Capability: Uses DuckDuckGo to search the web when the required information is not available locally and then extracts the relevant data.

Interactive UI: Uses the Streamlit library to create an interactive user interface that includes configuration options, chat history, and data loading status updates.

Code Explanation
Setup and Configuration
The application begins by importing the necessary libraries and initializing Streamlit session state variables. These variables store the chat history, bond details, cash flow information, company insights, and data loading status. Users can input their GROQ API key, provide Google Drive URLs for data files, and configure AI model settings (temperature, max tokens) via the sidebar.

Data Loading
The application loads data from Google Drive using the following steps:

get_file_id_from_url: Extracts the file ID from a Google Drive URL.

load_csv_from_drive_url: Downloads and reads CSV files from Google Drive using the gdown library.

validate_csv_file: Verifies that the CSV files have the expected columns.

load_data_from_drive: Manages the loading of bond, cash flow, and company data. The app fixes case-sensitive column names automatically.

Query Processing
When a user enters a query, the application:

Analyzes the query to determine what type of information is being requested.

Extracts key information such as ISIN numbers, prices, or company names.

Retrieves relevant data based on the identified query type.

Formats the data for the language model.

The application has the ability to process bond queries by ISIN, find cash flow schedules, perform yield calculations using bond coupon rates and prices, and search for company information.

Response Generation
The application generates responses by:

Creating a context object that contains all of the relevant information.

Passing the context, along with a prompt template, to the language model.

Formatting the response using Markdown.

Adding the response to the chat history.

Web Interface
The main user interface displays:

A welcome message with example queries.

Data loading status indicators.

A chat input field and the chat history.

Debug information (optional).

The application maintains a conversation history, allowing users to view past interaction
