import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Dict, List, Any, Tuple, Optional
import json
import tempfile
import os
import time
import io
import logging
import requests
from bs4 import BeautifulSoup
import gdown
import re

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="Tap Bonds AI Chatbot", page_icon="📊")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bond_details" not in st.session_state:
    st.session_state.bond_details = None
if "cashflow_details" not in st.session_state:
    st.session_state.cashflow_details = None
if "company_insights" not in st.session_state:
    st.session_state.company_insights = None
if "data_loading_status" not in st.session_state:
    st.session_state.data_loading_status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"}
    }
if "last_load_attempt" not in st.session_state:
    st.session_state.last_load_attempt = 0
if "search_results" not in st.session_state:
    st.session_state.search_results = {}
if "web_search_cache" not in st.session_state:
    st.session_state.web_search_cache = {}

def get_file_id_from_url(url):
    """Extract file ID from Google Drive sharing URL"""
    if not url:
        return None
    try:
        if '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
        elif '/open?id=' in url:
            return url.split('/open?id=')[1].split('&')[0]
        else:
            return url.split('/')[-2]
    except Exception:
        return None

def load_csv_from_drive_url(url):
    """Load CSV from Google Drive sharing URL using gdown"""
    file_id = get_file_id_from_url(url)
    if not file_id:
        return None, f"Invalid URL format: {url}"
    
    try:
        # Create a temporary file to download to
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
        
        # Download the file using gdown
        download_url = f'https://drive.google.com/uc?id={file_id}'
        st.write(f"Attempting to download from: {download_url}")
        gdown.download(download_url, temp_path, quiet=False)
        
        # Check if file exists and has content
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            st.write(f"File downloaded successfully, size: {os.path.getsize(temp_path)} bytes")
        else:
            st.write("File download failed or file is empty")
            return None, "Downloaded file is empty or missing"
            
        # Read the CSV file
        df = pd.read_csv(temp_path)
        
        # Display column information for debugging
        st.write(f"Columns in downloaded file: {list(df.columns)}")
        
        # Clean up
        os.unlink(temp_path)
        
        if df.empty:
            return None, "Downloaded file is empty"
            
        # Normalize column names (lowercase for case-insensitive comparison)
        df.columns = [col.lower() for col in df.columns]
        
        return df, "Success"
    except Exception as e:
        # Clean up in case of error
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        return None, f"Error downloading or parsing file: {str(e)}"

def validate_csv_file(df, expected_columns):
    """Validate if DataFrame has the expected columns (case-insensitive)"""
    if df is None:
        return False, "DataFrame is None"
    if df.empty:
        return False, "DataFrame is empty"
    
    # Convert expected columns to lowercase for case-insensitive comparison
    expected_columns_lower = [col.lower() for col in expected_columns]
    df_columns_lower = [col.lower() for col in df.columns]
    
    missing_columns = [col for col in expected_columns_lower if col not in df_columns_lower]
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    return True, "DataFrame validated successfully"

def load_data_from_drive(bond_urls, cashflow_url, company_url):
    """Load data from Google Drive URLs with improved error handling"""
    bond_details, cashflow_details, company_insights = None, None, None
    status = {
        "bond": {"status": "not_started", "message": ""}, 
        "cashflow": {"status": "not_started", "message": ""}, 
        "company": {"status": "not_started", "message": ""}
    }
    
    # Load bond files
    if bond_urls and any(bond_urls):
        status["bond"]["status"] = "in_progress"
        bond_dfs = []
        
        for i, url in enumerate(bond_urls):
            if not url:
                continue
                
            # Log attempt to load file
            st.write(f"Attempting to load bond file {i+1}...")
            
            df, message = load_csv_from_drive_url(url)
            
            if df is not None:
                # Check for both 'isin' and 'ISIN' columns (case-insensitive)
                is_valid, validation_message = validate_csv_file(df, ['isin', 'company_name'])
                
                if is_valid:
                    st.write(f"Successfully loaded bond file {i+1} with {len(df)} records")
                    bond_dfs.append(df)
                else:
                    st.write(f"Bond file {i+1} validation failed: {validation_message}")
                    # Try to fix common issues
                    if "Missing columns" in validation_message:
                        # Check if columns exist with different capitalization
                        if 'ISIN' in df.columns and 'isin' not in df.columns:
                            df = df.rename(columns={'ISIN': 'isin'})
                            st.write("Fixed: Renamed 'ISIN' to 'isin'")
                        if 'COMPANY_NAME' in df.columns and 'company_name' not in df.columns:
                            df = df.rename(columns={'COMPANY_NAME': 'company_name'})
                            st.write("Fixed: Renamed 'COMPANY_NAME' to 'company_name'")
                        
                        # Check again after fixes
                        is_valid, validation_message = validate_csv_file(df, ['isin', 'company_name'])
                        if is_valid:
                            st.write(f"Successfully fixed and loaded bond file {i+1} with {len(df)} records")
                            bond_dfs.append(df)
                        else:
                            status["bond"]["status"] = "error"
                            status["bond"]["message"] = f"Bond file {i+1}: {validation_message}"
                    else:
                        status["bond"]["status"] = "error"
                        status["bond"]["message"] = f"Bond file {i+1}: {validation_message}"
            else:
                st.write(f"Failed to load bond file {i+1}: {message}")
                status["bond"]["status"] = "error"
                status["bond"]["message"] = f"Error reading bond file {i+1}: {message}"
        
        if bond_dfs:
            try:
                bond_details = pd.concat(bond_dfs, ignore_index=True)
                bond_details = bond_details.drop_duplicates(subset=['isin'], keep='first')
                status["bond"]["status"] = "success"
                status["bond"]["message"] = f"Loaded {len(bond_details)} bonds"
                st.write(f"Successfully concatenated {len(bond_dfs)} bond files with {len(bond_details)} total records")
            except Exception as e:
                status["bond"]["status"] = "error"
                status["bond"]["message"] = f"Error concatenating bond data: {str(e)}"
                st.write(f"Error concatenating bond files: {str(e)}")
        elif status["bond"]["status"] != "error":
            status["bond"]["status"] = "error"
            status["bond"]["message"] = "No valid bond files processed"
    else:
        status["bond"]["status"] = "not_started"
        status["bond"]["message"] = "No bond URLs provided"
        
    # Load cashflow data
    if cashflow_url:
        status["cashflow"]["status"] = "in_progress"
        
        cashflow_details, message = load_csv_from_drive_url(cashflow_url)
        
        if cashflow_details is not None:
            is_valid, validation_message = validate_csv_file(
                cashflow_details, ['isin', 'cash_flow_date', 'cash_flow_amount'])
            
            if is_valid:
                status["cashflow"]["status"] = "success"
                status["cashflow"]["message"] = f"Loaded {len(cashflow_details)} cashflow records"
            else:
                # Try to fix common issues
                if "Missing columns" in validation_message:
                    # Check if columns exist with different capitalization
                    if 'ISIN' in cashflow_details.columns and 'isin' not in cashflow_details.columns:
                        cashflow_details = cashflow_details.rename(columns={'ISIN': 'isin'})
                    if 'CASH_FLOW_DATE' in cashflow_details.columns and 'cash_flow_date' not in cashflow_details.columns:
                        cashflow_details = cashflow_details.rename(columns={'CASH_FLOW_DATE': 'cash_flow_date'})
                    if 'CASH_FLOW_AMOUNT' in cashflow_details.columns and 'cash_flow_amount' not in cashflow_details.columns:
                        cashflow_details = cashflow_details.rename(columns={'CASH_FLOW_AMOUNT': 'cash_flow_amount'})
                    
                    # Check again after fixes
                    is_valid, validation_message = validate_csv_file(
                        cashflow_details, ['isin', 'cash_flow_date', 'cash_flow_amount'])
                    if is_valid:
                        status["cashflow"]["status"] = "success"
                        status["cashflow"]["message"] = f"Loaded {len(cashflow_details)} cashflow records"
                    else:
                        status["cashflow"]["status"] = "error"
                        status["cashflow"]["message"] = validation_message
                        cashflow_details = None
                else:
                    status["cashflow"]["status"] = "error"
                    status["cashflow"]["message"] = validation_message
                    cashflow_details = None
        else:
            status["cashflow"]["status"] = "error"
            status["cashflow"]["message"] = f"Error reading cashflow file: {message}"
    else:
        status["cashflow"]["status"] = "not_started"
        status["cashflow"]["message"] = "No cashflow URL provided"
        
    # Load company data
    if company_url:
        status["company"]["status"] = "in_progress"
        
        company_insights, message = load_csv_from_drive_url(company_url)
        
        if company_insights is not None:
            is_valid, validation_message = validate_csv_file(company_insights, ['company_name'])
            
            if is_valid:
                status["company"]["status"] = "success"
                status["company"]["message"] = f"Loaded {len(company_insights)} company records"
            else:
                # Try to fix common issues
                if "Missing columns" in validation_message:
                    # Check if columns exist with different capitalization
                    if 'COMPANY_NAME' in company_insights.columns and 'company_name' not in company_insights.columns:
                        company_insights = company_insights.rename(columns={'COMPANY_NAME': 'company_name'})
                    
                    # Check again after fixes
                    is_valid, validation_message = validate_csv_file(company_insights, ['company_name'])
                    if is_valid:
                        status["company"]["status"] = "success"
                        status["company"]["message"] = f"Loaded {len(company_insights)} company records"
                    else:
                        status["company"]["status"] = "error"
                        status["company"]["message"] = validation_message
                        company_insights = None
                else:
                    status["company"]["status"] = "error"
                    status["company"]["message"] = validation_message
                    company_insights = None
        else:
            status["company"]["status"] = "error"
            status["company"]["message"] = f"Error reading company file: {message}"
    else:
        status["company"]["status"] = "not_started"
        status["company"]["message"] = "No company URL provided"
    
    return bond_details, cashflow_details, company_insights, status

def get_llm(api_key, model_option, temperature, max_tokens):
    if not api_key:
        return None
    try:
        return ChatGroq(
            model=model_option,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception:
        return None

def perform_web_search(query, num_results=3):
    try:
        # Check if we have cached results
        if query in st.session_state.web_search_cache:
            return st.session_state.web_search_cache[query]
            
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, num_results)
        
        # Cache results
        st.session_state.web_search_cache[query] = results
        return results
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return []

def scrape_webpage(url):
    """Scrape content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit text length
        return text[:5000]
    except Exception as e:
        logger.error(f"Error scraping webpage {url}: {str(e)}")
        return f"Error scraping webpage: {str(e)}"

# Data access functions
def get_bond_details(bond_details, isin=None):
    if bond_details is None:
        return {"error": "Bond data not loaded"}
    
    if isin and isin in bond_details['isin'].values:
        row = bond_details[bond_details['isin'] == isin].iloc[0].to_dict()
        # Parse JSON fields
        for field in ['coupon_details', 'issuer_details', 'instrument_details']:
            try:
                if field in row and isinstance(row[field], str):
                    row[field] = json.loads(row[field])
            except Exception:
                row[field] = {"error": "Failed to parse JSON data"}
        return row
    return {"error": f"Bond with ISIN {isin} not found"}

def search_bond_by_text(bond_details, text):
    if bond_details is None:
        return [{"error": "Bond data not loaded"}]
    
    results = []
    for _, row in bond_details.iterrows():
        if (text.lower() in str(row.get('isin', '')).lower() or 
            text.lower() in str(row.get('company_name', '')).lower()):
            results.append(row.to_dict())
    
    if not results:
        return [{"error": f"No bonds found matching '{text}'"}]
    return results

def get_cashflow(cashflow_details, isin):
    if cashflow_details is None:
        return [{"error": "Cashflow data not loaded"}]
    
    if isin:
        cf_data = cashflow_details[cashflow_details['isin'] == isin]
        if not cf_data.empty:
            return cf_data.to_dict('records')
        return [{"error": f"No cashflow data for ISIN {isin}"}]
    return [{"error": "No ISIN provided"}]

def search_company(company_insights, company_name):
    if company_insights is None:
        return {"error": "Company data not loaded"}
    
    if company_name:
        matches = company_insights[company_insights['company_name'].str.contains(
            company_name, case=False, na=False)]
        if not matches.empty:
            company_data = matches.iloc[0].to_dict()
            # Parse JSON fields
            for field in ['key_metrics', 'income_statement', 'balance_sheet', 'cashflow']:
                try:
                    if field in company_data and isinstance(company_data[field], str):
                        company_data[field] = json.loads(company_data[field])
                except Exception:
                    company_data[field] = {}
            return company_data
        return {"error": f"No company found matching '{company_name}'"}
    return {"error": "No company name provided"}

def calculate_yield(bond_details, isin, price=None):
    bond_data = get_bond_details(bond_details, isin)
    if "error" in bond_data:
        return bond_data
    
    if price is None:
        return {"bond": bond_data, "error": "No price provided for yield calculation"}
    
    # Get coupon rate
    coupon_rate = 0
    if isinstance(bond_data.get('coupon_details'), dict):
        coupon_rate = float(bond_data['coupon_details'].get('rate', 0))
    
    # Simple yield calculation
    simple_yield = (coupon_rate / price) * 100
    return {
        "bond": bond_data,
        "price": price,
        "yield": round(simple_yield, 2)
    }

def extract_isin(text):
    """Extract ISIN from text - improved version"""
    # Standard ISIN pattern for Indian bonds starting with INE
    isin_pattern = r'\b(INE[A-Z0-9]{9})\b'
    matches = re.findall(isin_pattern, text.upper())
    if matches:
        return matches[0]
    
    # Look for any word that might be an ISIN
    for word in text.split():
        clean_word = ''.join(c for c in word if c.isalnum() or c in ".")
        if clean_word.upper().startswith("INE") and len(clean_word) >= 10:
            return clean_word.upper()
    
    return None

def process_query(query, bond_details, cashflow_details, company_insights):
    # First, check if data is loaded
    data_loaded = (bond_details is not None or 
                  cashflow_details is not None or 
                  company_insights is not None)
    
    # Extract ISIN if present
    isin = extract_isin(query)
    
    # Extract price if present
    price = None
    price_pattern = r'\$(\d+\.?\d*)'
    price_matches = re.findall(price_pattern, query)
    if price_matches:
        try:
            price = float(price_matches[0])
        except ValueError:
            pass
    
    # Extract company name
    company_name = None
    if "company" in query.lower():
        parts = query.lower().split("company")
        if len(parts) > 1 and len(parts[1].strip()) > 0:
            company_name = parts[1].strip()
    
    # Determine query type with stronger prioritization of local data
    query_type = "unknown"
    query_lower = query.lower()
    
    # Prioritize local data types over web search
    if "cash flow" in query_lower or "cashflow" in query_lower:
        query_type = "cashflow"
    elif "yield" in query_lower or "calculate" in query_lower:
        query_type = "yield"
    elif "company" in query_lower or "issuer" in query_lower:
        query_type = "company"
    elif isin or "isin" in query_lower or "details" in query_lower or "show" in query_lower:
        query_type = "bond"
    # Only use web search if explicitly requested or no local data
    elif ("search" in query_lower or "find" in query_lower or "web" in query_lower) or not data_loaded:
        query_type = "web_search"
    
    # Prepare context
    context = {
        "query": query,
        "query_type": query_type,
        "isin": isin,
        "company_name": company_name,
        "price": price
    }
    
    # Check for data availability
    if bond_details is None and query_type in ["bond", "yield"]:
        context["data_status"] = "Bond data not loaded. Please upload bond files."
    if cashflow_details is None and query_type == "cashflow":
        context["data_status"] = "Cashflow data not loaded. Please upload cashflow file."
    if company_insights is None and query_type == "company":
        context["data_status"] = "Company data not loaded. Please upload company insights file."

    return context

def display_status_indicator(status):
    """Use ASCII-compatible status indicators"""
    if status == "success":
        return "[SUCCESS]"  # Instead of "✅"
    elif status == "error":
        return "[ERROR]"    # Instead of "❌"
    elif status == "in_progress":
        return "[IN PROGRESS]"  # Instead of "⏳"
    else:
        return "[NOT STARTED]"  # Instead of "⚪"

def process_web_search_query(query):
    # Extract search terms
    search_terms = query.lower().replace("search", "").replace("web", "").replace("for", "").strip()
    if not search_terms:
        search_terms = "bond market news"
        
    # Perform search
    results = perform_web_search(search_terms)
    
    # Also try to scrape the first result for more context
    if results and len(results) > 0 and 'link' in results[0]:
        content = scrape_webpage(results[0]['link'])
        return {"search_results": results, "scraped_content": content}
    
    return {"search_results": results}

def process_bond_query(query, bond_details, isin=None):
    if isin:
        return get_bond_details(bond_details, isin)
        
    # Try to extract search terms
    search_terms = query.lower().replace("bond", "").replace("details", "").replace("about", "").replace("show", "").strip()
    if search_terms:
        return {"search_results": search_bond_by_text(bond_details, search_terms)}
        
    return {"error": "Please provide an ISIN or search terms to find bond details"}

def process_cashflow_query(query, cashflow_details, isin=None):
    if isin:
        return {"cashflow": get_cashflow(cashflow_details, isin)}
        
    # Try to extract search terms
    search_terms = query.lower().replace("cashflow", "").replace("cash flow", "").replace("for", "").strip()
    if search_terms:
        # Try to find an ISIN in the search terms
        extracted_isin = extract_isin(search_terms)
        if extracted_isin:
            return {"cashflow": get_cashflow(cashflow_details, extracted_isin)}
                
    return {"error": "Please provide an ISIN to find cashflow details"}

def process_company_query(query, company_insights, company_name=None):
    if company_name:
        return search_company(company_insights, company_name)
        
    # Try to extract company name
    search_terms = query.lower().replace("company", "").replace("issuer", "").replace("about", "").strip()
    if search_terms:
        return search_company(company_insights, search_terms)
        
    return {"error": "Please provide a company name to find company details"}

def process_yield_query(query, bond_details, isin=None, price=None):
    if isin and price:
        return calculate_yield(bond_details, isin, price)
        
    # Try to extract price from query if ISIN exists
    if isin and not price:
        price_pattern = r'\$(\d+\.?\d*)'
        price_matches = re.findall(price_pattern, query)
        if price_matches:
            try:
                price = float(price_matches[0])
                return calculate_yield(bond_details, isin, price)
            except ValueError:
                pass
                    
    # Return error if missing parameters
    if not isin:
        return {"error": "Please provide an ISIN to calculate yield"}
    if not price:
        return {"error": "Please provide a price to calculate yield"}
        
    return {"error": "Could not process yield calculation"}

def format_bond_details_response(bond_data):
    """Format bond details according to sample responses"""
    if "error" in bond_data:
        return f"Error: {bond_data['error']}"
    
    response = f"## Bond Details for {bond_data.get('isin', 'Unknown ISIN')}\n\n"
    
    # Basic bond information
    response += f"**Issuer**: {bond_data.get('company_name', 'N/A')}\n"
    
    # Coupon details
    coupon_details = bond_data.get('coupon_details', {})
    if isinstance(coupon_details, dict):
        response += f"**Coupon Rate**: {coupon_details.get('rate', 'N/A')}%\n"
        response += f"**Coupon Type**: {coupon_details.get('type', 'N/A')}\n"
        response += f"**Coupon Frequency**: {coupon_details.get('frequency', 'N/A')}\n"
    
    # Instrument details
    instrument_details = bond_data.get('instrument_details', {})
    if isinstance(instrument_details, dict):
        response += f"**Issue Date**: {instrument_details.get('issue_date', 'N/A')}\n"
        response += f"**Maturity Date**: {instrument_details.get('maturity_date', 'N/A')}\n"
        response += f"**Issue Size**: ₹{instrument_details.get('issue_size', 'N/A')}\n"
        response += f"**Face Value**: ₹{instrument_details.get('face_value', 'N/A')}\n"
    
    # Issuer details
    issuer_details = bond_data.get('issuer_details', {})
    if isinstance(issuer_details, dict):
        response += f"**Sector**: {issuer_details.get('sector', 'N/A')}\n"
        response += f"**Industry**: {issuer_details.get('industry', 'N/A')}\n"
        response += f"**Credit Rating**: {issuer_details.get('credit_rating', 'N/A')}\n"
    
    return response

def format_cashflow_response(cashflow_data, isin):
    """Format cashflow response according to sample responses"""
    if isinstance(cashflow_data, list) and "error" in cashflow_data[0]:
        return f"Error: {cashflow_data[0]['error']}"
    
    response = f"## Cash Flow Schedule for {isin}\n\n"
    
    if not cashflow_data or len(cashflow_data) == 0:
        return response + "No cash flow data available for this bond."
    
    # Create a table header
    response += "| Cash Flow Date | Cash Flow Amount | Record Date | Principal Amount | Interest Amount | TDS Amount | Remaining Principal |\n"
    response += "|---------------|-----------------|-------------|-------------------|----------------|------------|---------------------|\n"
    
    # Add each cash flow as a row
    for cf in cashflow_data:
        response += f"| {cf.get('cash_flow_date', 'N/A')} "
        response += f"| {cf.get('cash_flow_amount', 'N/A')} "
        response += f"| {cf.get('record_date', 'N/A')} "
        response += f"| {cf.get('principal_amount', 0.0)} "
        response += f"| {cf.get('interest_amount', 'N/A')} "
        response += f"| {cf.get('tds_amount', 'N/A')} "
        response += f"| {cf.get('remaining_principal', 'N/A')} |\n"
    
    response += "\nPlease note that this cash flow schedule is based on the provided data and may not reflect the actual cash flows of the bond."
    
    return response

def format_yield_response(yield_data):
    """Format yield calculation response according to sample responses"""
    if "error" in yield_data:
        return f"Error: {yield_data['error']}"
    
    bond_data = yield_data.get("bond", {})
    price = yield_data.get("price", 0)
    yield_value = yield_data.get("yield", 0)
    
    response = f"## Yield Calculation for {bond_data.get('isin', 'Unknown ISIN')}\n\n"
    
    response += f"**Bond**: {bond_data.get('company_name', 'N/A')} ({bond_data.get('isin', 'N/A')})\n"
    
    # Coupon details
    coupon_details = bond_data.get('coupon_details', {})
    if isinstance(coupon_details, dict):
        response += f"**Coupon Rate**: {coupon_details.get('rate', 'N/A')}%\n"
    
    response += f"**Price**: ${price}\n"
    response += f"**Calculated Yield**: {yield_value}%\n\n"
    
    response += "The yield is calculated using the simple formula: (Coupon Rate / Price) * 100\n"
    response += "For a more accurate yield calculation, please consider using a financial calculator that accounts for time value of money."
    
    return response

def format_company_response(company_data):
    """Format company information response according to sample responses"""
    if "error" in company_data:
        return f"Error: {company_data['error']}"
    
    response = f"## Company Information: {company_data.get('company_name', 'Unknown Company')}\n\n"
    
    # Key metrics
    key_metrics = company_data.get('key_metrics', {})
    if isinstance(key_metrics, dict) and key_metrics:
        response += "### Key Financial Metrics\n\n"
        for key, value in key_metrics.items():
            response += f"**{key.replace('_', ' ').title()}**: {value}\n"
        response += "\n"
    
    # Income statement
    income_statement = company_data.get('income_statement', {})
    if isinstance(income_statement, dict) and income_statement:
        response += "### Income Statement Highlights\n\n"
        for key, value in income_statement.items():
            response += f"**{key.replace('_', ' ').title()}**: ₹{value}\n"
        response += "\n"
    
    # Balance sheet
    balance_sheet = company_data.get('balance_sheet', {})
    if isinstance(balance_sheet, dict) and balance_sheet:
        response += "### Balance Sheet Highlights\n\n"
        for key, value in balance_sheet.items():
            response += f"**{key.replace('_', ' ').title()}**: ₹{value}\n"
        response += "\n"
    
    # Cash flow
    cashflow = company_data.get('cashflow', {})
    if isinstance(cashflow, dict) and cashflow:
        response += "### Cash Flow Highlights\n\n"
        for key, value in cashflow.items():
            response += f"**{key.replace('_', ' ').title()}**: ₹{value}\n"
    
    return response

def format_web_search_response(search_data):
    """Format web search response"""
    search_results = search_data.get("search_results", [])
    scraped_content = search_data.get("scraped_content", "")
    
    if not search_results:
        return "No web search results found. Please try a different query."
    
    response = "## Web Search Results\n\n"
    
    for i, result in enumerate(search_results[:3]):
        response += f"### {i+1}. {result.get('title', 'No Title')}\n"
        response += f"**Source**: {result.get('link', 'No Link')}\n"
        response += f"{result.get('snippet', 'No snippet available')}\n\n"
    
    if scraped_content:
        response += "### Additional Information\n"
        response += scraped_content[:500] + "...\n\n"
    
    response += "Please note that these results are from a web search and may not be specific to the Tap Bonds platform."
    
    return response

def generate_response(context, llm):
    """Generate a response based on context and query type"""
    if llm is None:
        return "Please enter a valid GROQ API key in the sidebar to continue."
    
    if "error" in context:
        return f"Error: {context['error']}"
        
    if "data_status" in context:
        return f"I need more data to answer your question. {context['data_status']}"
    
    # Format response based on query type
    query_type = context.get("query_type", "unknown")
    
    # Bond details response
    if query_type == "bond":
        if "search_results" in context:
            bonds = context["search_results"]
            if len(bonds) == 1:
                return format_bond_details_response(bonds[0])
            else:
                response = "## Bond Search Results\n\n"
                for i, bond in enumerate(bonds[:5]):
                    if "error" in bond:
                        response += f"{bond['error']}\n"
                        continue
                    response += f"### Bond {i+1}: {bond.get('company_name', 'N/A')}\n"
                    response += f"**ISIN**: {bond.get('isin', 'N/A')}\n"
                    if 'coupon_details' in bond and isinstance(bond['coupon_details'], dict):
                        response += f"**Coupon Rate**: {bond['coupon_details'].get('rate', 'N/A')}%\n"
                    response += "---\n"
                return response
        else:
            return format_bond_details_response(context)
    
    # Cash flow response
    elif query_type == "cashflow" and "cashflow" in context:
        return format_cashflow_response(context["cashflow"], context.get("isin", "Unknown ISIN"))
    
    # Yield calculation response
    elif query_type == "yield":
        return format_yield_response(context)
    
    # Company information response
    elif query_type == "company":
        return format_company_response(context)
    
    # Web search response
    elif query_type == "web_search" and "search_results" in context:
        return format_web_search_response(context)
    
    # Default response using LLM
    template = """You are a helpful financial assistant specializing in bonds.
    User Query: {query}
    Query Type: {query_type}
    
    Available Context:
    {context_str}
    
    Respond in a professional, friendly manner with Markdown formatting.
    If you cannot answer from the provided data, politely say so.
    """
    
    # Convert context to a formatted string
    context_parts = []
    for key, value in context.items():
        if key not in ["query", "query_type"] and value:
            context_parts.append(f"{key}: {value}")
    
    context_str = "\n".join(context_parts)
    
    # Create and run the chain
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": context["query"], "query_type": context["query_type"], "context_str": context_str})

def main():
    """Main application function"""
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input("Enter your GROQ API Key", type="password")
        
        st.markdown("### Google Drive Integration")
        
        # File URL inputs
        st.markdown("#### Bond Detail Files URLs")
        bond_urls = []
        for i in range(4):  # Allow up to 4 bond file parts
            bond_url = st.text_input(
                f"Bond Details CSV Part {i+1} URL", 
                key=f"bond_url_{i}",
                help="Enter Google Drive sharing URL"
            )
            bond_urls.append(bond_url)
        
        cashflow_url = st.text_input(
            "Cashflow Details CSV URL",
            help="Enter Google Drive sharing URL"
        )
        
        company_url = st.text_input(
            "Company Insights CSV URL",
            help="Enter Google Drive sharing URL"
        )
        
        if st.button("Load Data from Drive"):
            with st.spinner("Loading data from Google Drive..."):
                st.session_state.last_load_attempt = time.time()
                st.session_state.bond_details, st.session_state.cashflow_details, st.session_state.company_insights, st.session_state.data_loading_status = load_data_from_drive(
                    bond_urls, cashflow_url, company_url
                )
                
                if (st.session_state.bond_details is not None or 
                    st.session_state.cashflow_details is not None or 
                    st.session_state.company_insights is not None):
                    st.success("Data loaded successfully from Google Drive!")
                else:
                    st.error("Failed to load data from Google Drive. Check the Debug Information.")
        
        # Model configuration
        st.markdown("### Model Configuration")
        model_option = st.selectbox(
            "Select Model",
            ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
        )
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=500, max_value=4000, value=1500, step=500)
    
    # Main content
    st.title("Tap Bonds AI Chatbot")
    st.markdown("""
    Welcome to the Tap Bonds AI Chatbot! 💼🔍

    Ask about bonds, companies, cash flows, yields, or search web for more information.

    **Example queries:**  
    - "Show details for INE08XP07258"  
    - "What's the cash flow schedule for INE08XP07258?"  
    - "Calculate yield for INE08XP07258 at $96.50"
    - "Search web for recent Indian bond market trends"
    """)

    # Data status section
    st.markdown("### Data Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bond_status = st.session_state.data_loading_status.get("bond") or {"status": "not_started", "message": "Not loaded"}
        st.markdown(f"{display_status_indicator(bond_status['status'])} **Bond Data:** {bond_status['message']}")
        
    with col2:
        cashflow_status = st.session_state.data_loading_status.get("cashflow") or {"status": "not_started", "message": "Not loaded"}
        st.markdown(f"{display_status_indicator(cashflow_status['status'])} **Cashflow Data:** {cashflow_status['message']}")
        
    with col3:
        company_status = st.session_state.data_loading_status.get("company") or {"status": "not_started", "message": "Not loaded"}
        st.markdown(f"{display_status_indicator(company_status['status'])} **Company Data:** {company_status['message']}")

    # Check for API key
    if not api_key:
        st.warning("⚠️ Please enter your GROQ API key in the sidebar to interact with the chatbot.")

    st.markdown("---")

    # Create columns for input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Enter your query:", key="query_input")
    with col2:
        submit_button = st.button("Submit", use_container_width=True)

    # Process query when Submit is clicked
    if submit_button and query:
        # Initialize LLM only when needed
        llm = get_llm(api_key, model_option, temperature, max_tokens)
        
        with st.spinner("Processing your query..."):
            # Add user query to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Process query
            context = process_query(
                query, 
                st.session_state.bond_details,
                st.session_state.cashflow_details,
                st.session_state.company_insights
            )
            
            # Handle different query types
            query_type = context.get("query_type", "unknown")
            
            # Perform specific processing based on query type
            if query_type == "bond":
                bond_result = process_bond_query(
                    query, 
                    st.session_state.bond_details, 
                    context.get("isin")
                )
                context.update(bond_result)
            elif query_type == "cashflow":
                cashflow_result = process_cashflow_query(
                    query, 
                    st.session_state.cashflow_details, 
                    context.get("isin")
                )
                context.update(cashflow_result)
            elif query_type == "company":
                company_result = process_company_query(
                    query, 
                    st.session_state.company_insights, 
                    context.get("company_name")
                )
                context.update(company_result)
            elif query_type == "yield":
                yield_result = process_yield_query(
                    query, 
                    st.session_state.bond_details, 
                    context.get("isin"), 
                    context.get("price")
                )
                context.update(yield_result)
            elif query_type == "web_search":
                search_results = process_web_search_query(query)
                context.update(search_results)
            
            response = generate_response(context, llm)
            
            # Add bot response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history in reverse order (newest first)
    st.markdown("### Conversation")
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"**You**: {message['content']}")
        else:
            st.markdown(f"**Tap Bonds AI**: {message['content']}")

    # Add footer
    st.markdown("---")
    st.markdown("Powered by Tap Bonds AI")

    # Display debugging information in an expander
    with st.expander("Debug Information", expanded=False):
        st.write("Data Availability:")
        st.write(f"- Bond Details: {display_status_indicator(st.session_state.data_loading_status['bond']['status'])} {st.session_state.data_loading_status['bond']['message']}")
        st.write(f"- Cashflow Details: {display_status_indicator(st.session_state.data_loading_status['cashflow']['status'])} {st.session_state.data_loading_status['cashflow']['message']}")
        st.write(f"- Company Insights: {display_status_indicator(st.session_state.data_loading_status['company']['status'])} {st.session_state.data_loading_status['company']['message']}")
        
        # Add data sample viewer
        if st.checkbox("Show Data Samples"):
            if st.session_state.bond_details is not None:
                st.subheader("Bond Details Sample")
                st.dataframe(st.session_state.bond_details.head(3))
            
            if st.session_state.cashflow_details is not None:
                st.subheader("Cashflow Details Sample")
                st.dataframe(st.session_state.cashflow_details.head(3))
            
            if st.session_state.company_insights is not None:
                st.subheader("Company Insights Sample")
                st.dataframe(st.session_state.company_insights.head(3))

if __name__ == "__main__":
    main()
