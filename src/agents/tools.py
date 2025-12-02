import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
import os
from typing import List
from tavily import TavilyClient
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


def internet_search(query: str, max_results: int = 5) -> str:
    """
    Search the internet for stock information.
    Use for: stock discovery, competitive analysis, news, catalysts
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        String representation of search results
    """
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    results = tavily_client.search(query, max_results=max_results, include_raw_content=True)
    return str(results)


def get_cagr(ticker_list: List[str], period_years: int = 10) -> str:
    """
    Downloads historical stock data for the specified period, calculates the 
    Cumulative Annual Growth Rate (CAGR) for each ticker, and returns the results.

    Args:
        ticker_list: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        period_years: The length of the historical period to analyze (default: 10)
    
    Returns:
        String representation of a Pandas Series containing the CAGR (as a percentage) for each ticker
    """
    # 1. Define the Date Range
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * period_years)).strftime('%Y-%m-%d')
    
    # 2. Download Data
    try:
        data = yf.download(
            ticker_list,
            start=start_date,
            end=end_date,
            auto_adjust=True  # Ensure we use adjusted prices
        )['Close']
        
    except Exception as e:
        return f"Error fetching data for tickers {ticker_list}: {e}"

    # 3. Calculate CAGR
    
    # Check if data was retrieved successfully
    if data.empty:
        return "No data retrieved for the specified tickers"
    
    # Get the first and last prices
    start_prices = data.iloc[0]
    end_prices = data.iloc[-1]
    
    # Calculate the precise number of years
    num_days = (data.index[-1] - data.index[0]).days
    num_years = num_days / 365.25 
    
    # CAGR Formula: ((Ending Value / Beginning Value) ^ (1 / Number of Years)) - 1
    cagr_series = ((end_prices / start_prices) ** (1 / num_years)) - 1
    
    # Format the output for readability
    formatted_cagr = cagr_series.map('{:.2%}'.format)
    formatted_cagr.name = f'{period_years}-Year CAGR'
    
    return formatted_cagr.to_string()


def get_roe(ticker_list: List[str]) -> str:
    """
    Calculates the Return on Equity (ROE) for a list of tickers.
    
    Args:
        ticker_list: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
    
    Returns:
        String representation of ROE percentages for each ticker
    """
    roe_data = {}
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Get Net Income (from the latest annual income statement)
            income_stmt = stock.financials.transpose()
            net_income = income_stmt['Net Income'].iloc[0]

            # 2. Get Total Equity (from the latest annual balance sheet)
            balance_sheet = stock.balance_sheet.transpose()
            total_equity = balance_sheet['Total Equity Gross Minority Interest'].iloc[0]

            if total_equity != 0:
                roe = net_income / total_equity
                roe_data[ticker] = f"{roe * 100:.2f}%"
            else:
                roe_data[ticker] = 'N/A'

        except (KeyError, IndexError, Exception):
            roe_data[ticker] = 'N/A'
    
    result = pd.Series(roe_data, name='ROE')
    return result.to_string()


def get_debt_equity_ratio(ticker_list: List[str]) -> str:
    """
    Calculates the Debt-to-Equity Ratio for a list of tickers.
    
    Args:
        ticker_list: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
    
    Returns:
        String representation of Debt-to-Equity ratios for each ticker
    """
    de_data = {}
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            # Get the most recent balance sheet
            balance_sheet = stock.balance_sheet
            
            # Check if balance sheet data is available and non-empty
            if not balance_sheet.empty:
                # Find Total Debt and Total Equity
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                total_equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]
                
                if total_equity > 0:
                    de_ratio = total_debt / total_equity
                    de_data[ticker] = round(de_ratio, 2)
                else:
                    de_data[ticker] = 'Zero Equity'
            else:
                de_data[ticker] = 'No Data'

        except (KeyError, IndexError, Exception):
            # Handle cases where keys are missing or data retrieval fails
            de_data[ticker] = 'N/A'
    
    result = pd.Series(de_data, name='Debt/Equity Ratio')
    return result.to_string()


def get_pe_ratio(ticker_list: List[str]) -> str:
    """
    Fetches the Trailing P/E Ratio for a list of tickers.
    
    Args:
        ticker_list: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
    
    Returns:
        String representation of P/E ratios for each ticker
    """
    pe_data = {}
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            # Fetch the trailing P/E from the 'info' dictionary
            pe_ratio = stock.info.get('trailingPE')
            if pe_ratio is not None:
                pe_data[ticker] = round(pe_ratio, 2)
            else:
                pe_data[ticker] = 'N/A'
        except Exception:
            pe_data[ticker] = 'Error'
    
    result = pd.Series(pe_data, name='P/E Ratio')
    return result.to_string()
