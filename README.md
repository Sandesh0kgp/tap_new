
# Tap Bonds AI Chatbot Project

A Streamlit-based AI chatbot that leverages GROQ's language models to deliver bond market insights, cash flow analysis, and company information. The project integrates data validation from Google Drive CSV files and enhances query responses via the DuckDuckGo Search API.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture and Approach](#system-architecture-and-approach)
- [Setup and Configuration](#setup-and-configuration)
- [Data Loading](#data-loading)
- [Query Processing](#query-processing)
- [Response Generation](#response-generation)
- [Web Interface](#web-interface)
- [Conclusion](#conclusion)

---

## Overview

The Tap Bonds AI Chatbot is a web application designed using Streamlit to provide detailed analysis for bond markets. It handles tasks such as retrieving bond details, cash flow schedules, yield calculations, and company insights. In cases where local data is insufficient, it also performs web searches to supply up-to-date and relevant information.

---

## Key Features

- **AI-Driven Insights:** Utilizes GROQ's language models (e.g., Llama3 or Mixtral) for generating natural language responses.
- **Data Validation:** Implements comprehensive CSV data loading and validation from Google Drive.
- **Query Analysis:** Extracts key information from user queries to deliver accurate responses.
- **Web Search Integration:** Augments data retrieval with DuckDuckGo Search API for complex queries.
- **Interactive UI:** A user-friendly interface with real-time data loading indicators, configurable AI model settings, and an accessible chat history.

---

## System Architecture and Approach

The project follows a modular design comprising several key components:

| Component                | Description                                                                                                                                                                             |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Management**      | Loads bond, cash flow, and company data from CSV files on Google Drive.                                                                                                                 |
| **Query Processing**     | Analyzes user queries, extracts vital details (e.g., ISIN numbers, company names), and identifies the type of information needed (bond details, cash flows, yield calculations, etc.). |
| **LLM Integration**      | Integrates GROQ's language models to generate context-driven, natural language responses based on extracted query information.                                                          |
| **Web Search Capability**| Uses the DuckDuckGo Search API to scrape and parse web data when local information is not sufficient.                                                                                   |
| **Interactive UI**       | Implements an interactive interface using Streamlit, complete with chat history, data loading statuses, and configuration options.                                                       |

---

## Setup and Configuration

- **Library Imports:** The application begins by importing necessary libraries and initializing session state variables.
- **Session State Management:** Stores chat history, bond details, cash flow information, and company insights.
- **Configuration Options:** Users can input their GROQ API key, specify Google Drive URLs for CSV files, and adjust AI model settings (e.g., temperature, max tokens) via the sidebar.

---

## Data Loading

The project handles data loading from Google Drive CSV files in a multi-step process:

- **Extract File ID:** Uses `get_file_id_from_url` to extract the file ID from a Google Drive URL.
- **Download and Read CSV:** Employs `load_csv_from_drive_url` via the gdown library to download and read CSV files.
- **Validate CSV Content:** Uses `validate_csv_file` to ensure files have the required columns.
- **Manage Data Loading:** Integrates these functions in `load_data_from_drive` to automatically adjust case-sensitive columns and load bond, cash flow, and company data.

---

## Query Processing

When the user enters a query, the application:

- **Analyzes Query Intent:** Determines if the query pertains to bond details, cash flow, yield calculations, or company information.
- **Extracts Key Data:** Retrieves key pieces of information (such as ISIN numbers, prices, or company names).
- **Formats Data for AI:** Prepares the query and accompanying data for processing by the language model.
- **Specialized Actions:** Processes bond queries, identifies cash flow schedules, calculates yields using bond coupon rates and prices, and fetches company insights as needed.

---

## Response Generation

The chatbot generates responses using the following steps:

- **Context Creation:** Constructs a context object that aggregates all relevant data.
- **Language Model Invocation:** Passes the context and a predefined prompt template to the language model.
- **Formatting:** Generates and formats the response using Markdown, then appends it to the chat history.

---

## Web Interface

The chatbot’s user interface is built on Streamlit and includes:

- **Welcome Message:** An introductory message with example queries.
- **Data Loading Status:** Visual indicators to display the status of data uploads and validations.
- **Chat Interface:** A dynamic chat field paired with a history log for ongoing conversation tracking.
- **Debug Information:** Optional debug outputs to assist during development or troubleshooting.

---

## Conclusion

The Tap Bonds AI Chatbot Project is a comprehensive tool combining data management, intelligent query processing, and an interactive interface to provide valuable bond market analysis. Its modular design ensures scalability and ease of customization, making it a robust solution for real-time financial data insights.

