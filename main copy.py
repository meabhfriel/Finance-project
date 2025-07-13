from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests # Needed for the call_ollama function

# Import functions from your project.py and ai_engine_call.py
from project import fetch_stock_data, calculate_fundamental_indicators, dcf_parameters_calculator, generate_fundamental_dcf_prompt #module 2
from ai_engine_call import call_ollama #Module 1, NEEDS TO BE UPDATED TO CAOIMHES NEW FILE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is running"}


def build_prompt(user_preference: str, company_df: pd.DataFrame) -> str:
    companies = company_df.to_string(index=False)
    prompt = f"""
You are a financial AI assistant trained specifically to screen energy sector stocks based strictly on user-defined investment preferences.
Do not ask questions. Do not speculate. Use only the data provided. Always return results in a table.

User request:
"{user_preference}"

Company data:
{companies}

Instructions:
1. Evaluate each company based on relevant factors in the user request (0–5).
2. Score each factor, sum them up.
3. Show only the top 3 companies.
4. Return this table:

| Symbol | Company Name | Scores by Factor | Total Score | Rationale |
|--------|--------------|------------------|-------------|-----------|

Follow these constraints strictly. No paragraphs, no external data, no assumptions.
"""
    return prompt

# --- DCF Analysis Test Function ---
def run_dcf_analysis(ticker: str):
    print(f"\n--- Starting DCF Analysis for {ticker} ---")

    # 1. Fetch data
    ticker_info, hist_df, financial_statements_raw = fetch_stock_data(ticker)

    # Validate that fetching data was successful and financial_statements_raw is not None
    if ticker_info is None or financial_statements_raw is None:
        print(f"Failed to fetch data for {ticker} or financial statements are None. Aborting analysis.")
        return

    # financial_statements_raw is already a dictionary with the correct keys
    # 'income_statement', 'balance_sheet', 'cash_flow' from fetch_stock_data.
    # We can directly access them.
    financial_statements_fixed = {
        'income_statement': financial_statements_raw.get('income_statement'),
        'balance_sheet': financial_statements_raw.get('balance_sheet'),
        'cash_flow': financial_statements_raw.get('cash_flow')
    }

    # Verify that essential financial statements are present and not empty
    required_statements = ['income_statement', 'balance_sheet', 'cash_flow']
    for stmt_name in required_statements:
        df = financial_statements_fixed.get(stmt_name)
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"Error: '{stmt_name}' DataFrame for {ticker} is missing or empty. Aborting analysis.")
            return

    # 2. Calculate fundamental indicators
    df_with_fundamentals = calculate_fundamental_indicators(financial_statements_fixed)

    if df_with_fundamentals.empty:
        print(f"Could not calculate fundamental indicators for {ticker}. Aborting analysis.")
        return

    # 3. Calculate DCF parameters
    dcf_params = dcf_parameters_calculator(ticker_info, df_with_fundamentals)

    # 4. Generate AI prompt
    ai_dcf_prompt = generate_fundamental_dcf_prompt(ticker, df_with_fundamentals, dcf_params)

    print("\n--- Generated AI Prompt for DCF Analysis ---")
    print(ai_dcf_prompt)

    # 5. Call the LLM
    print("\nCalling LLM for analysis...")
    try:
        llm_response = call_ollama(ai_dcf_prompt)
        print("\nLLM Response")
        print(llm_response)
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to Ollama. Please ensure Ollama is running and the model ('gemma3') is available.")
        print("You might need to run `ollama run gemma3` in your terminal.")
    except Exception as e:
        print(f"\nAn error occurred while calling the LLM: {e}")

#FastAPI Endpoint (Existing from your file)
@app.post("/analyze")
async def analyze_stock(request: Request):
    data = await request.json()
    user_pref = data.get('preference', 'I want stable, US-based energy companies with solid growth.')

    # Example company data (can be replaced with reading from CSV or live data)
    energy_data = {
        "Symbol": ["OKE", "CVX", "KMI", "XOM", "BP"],
        "Company Name": ["Oneok", "Chevron Corporation", "Kinder Morgan", "ExxonMobil", "BP PLC"],
        "Ethics Score": [3, 4, 3, 3, 2],
        "Volatility Score": [4, 3, 3, 2, 4],
        "Growth Score": [4, 4, 4, 5, 3],
    }
    energy_df = pd.DataFrame(energy_data)

    # This part uses the `build_prompt` and calls an LLM,
    # distinct from the DCF analysis.
    prompt = build_prompt(user_pref, energy_df)
    llm_output = call_ollama(prompt) # Using call_ollama from ai_engine_call

    return {"reply": llm_output}

#Main Execution Block for Direct Testing
if __name__ == "__main__":
    # You can change this ticker to any stock you want to test the DCF analysis on
    test_ticker = 'MCD'
    run_dcf_analysis(test_ticker)

    # To run the FastAPI app, you would typically use:
    # uvicorn main:app --reload
    # This block is for direct script testing of the DCF analysis pipeline.