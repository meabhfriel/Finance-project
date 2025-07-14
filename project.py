import pandas as pd
import yfinance as yf
import numpy as np
from io import StringIO
selected_ticker = 'MCD' # <--- CHANGE THIS to the stock ticker you want to analyze 
all_companies_data = {}
print(f"Analyzing selected ticker: {selected_ticker}")
historical_prices_data = {}
financial_statements_data = {}


print(f"Fetching detailed data for {selected_ticker} via yfinance...")
try: #using try in case data not available for one of the tickers - so program wont crash, it will go to the except line
    stock = yf.Ticker(selected_ticker)

    # 1. Fetch Company Info (for Stock Screener)
    info = stock.info
    all_companies_data[selected_ticker] = {
        'Name': info.get('longName', 'N/A'), #returns 'N/A' if 'longname'/any other info doesn't exist
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'MarketCap': info.get('marketCap', 'N/A'),
        'PE_Ratio': info.get('trailingPE', 'N/A'),
        'DividendYield': info.get('dividendYield', 'N/A'),
        'RevenueGrowth': info.get('revenueGrowth', 'N/A'), # TTM revenue growth?
        'ProfitMargin': info.get('profitMargins', 'N/A'), # TTM profit margins?
        'DebtToEquity': info.get('debtToEquity', 'N/A'),
        'CurrentRatio': info.get('currentRatio', 'N/A'),
        'ReturnOnEquity': info.get('returnOnEquity', 'N/A'),
    }

        # 2. Stock Price data
    hist = stock.history(period="1y") #can change to a different period?
    historical_prices_data[selected_ticker] = hist

        
    financial_statements_data[selected_ticker] = {
        'Income Statement': stock.financials.T, #stock.financials retrieves the annual income statements for the ticker as a Pandas DataFrame. 
        'Balance Sheet': stock.balance_sheet.T, #stock.balance_sheet retrieves the annual balamce sheets
        'Cash Flow': stock.cashflow.T #stock.cashflow retrieves annual cashflow statements
    } #.T to transpose so that  dates are in rows and financial data in columns (to help make data easier to work with), and then we store these 3 financial statemen's info in financial_statements_data[ticker]


    print(f"Successfully fetched data for {selected_ticker}")

except Exception as e: #assigns any errors to variable e
    print(f"Could not fetch data for {selected_ticker}: {e}") #prints error for us

sp500_df_for_screener = pd.DataFrame.from_dict(all_companies_data, orient='index') #converts dictionary to pandas data frame (from #1 company info)

print("\nQuick summary of data")
print(f"\n{selected_ticker} Data (first 5 rows):")
print(sp500_df_for_screener.head())

print(f"\nHistorical Prices for {selected_ticker} (first 5 rows):")
if selected_ticker in historical_prices_data:
    print(historical_prices_data[selected_ticker].head())

print(f"\nHistorical Prices for {selected_ticker} (first 5 rows):")
if selected_ticker in historical_prices_data:
    print(historical_prices_data[selected_ticker].head())
example_stock_info = sp500_df_for_screener.loc[selected_ticker].to_string() #IS THIS STILL NEEDED WHEN WE CHANGE AI PACKAGE?


#some calculations below can be adjusted:
def calculate_technical_indicators_manual(df):
     
    if 'Close' not in df.columns: #checking that we even have close prices
        print("DataFrame must contain a 'Close' column for technical analysis.")
        return df.copy() # if no close prices it just returnes a copy of the original dataframe

    # working on a new object, so not to change original dataframe
    df_copy = df.copy()

    # Calculating moving averages:
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean() #sma - can change window if needed
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean() #lma - ""

    # Calculating Relative Strength Index (RSI) 
    window_length = 14
    delta = df_copy['Close'].diff() #difference between the current close and the previous close = daily price change
    gain = delta.where(delta > 0, 0) #This creates a Series where values are positive price changes (gains), and 0 otherwise.
    loss = -delta.where(delta < 0, 0) # This creates a Series where values are positive price losses (negative delta made positive), and 0 otherwise.

    avg_gain = gain.ewm(com=window_length - 1, min_periods=window_length).mean() #calculates the Exponential Weighted Moving Average (EWMA) of the gain values.
    avg_loss = loss.ewm(com=window_length - 1, min_periods=window_length).mean() #calculates the EWMA of the loss values

    rs = avg_gain / avg_loss #calculates the relative strength, which is the ratio of average gains to average losses - if avg loss=0, you will get Nan/Inf becuase of division by 0.
    df_copy['RSI'] = 100 - (100 / (1 + rs)) #RSI formula, assigned to new col 'RSI'

    # calculating Moving Average Convergence Divergence (MACD) 
    # MACD uses 12-period EMA, 26-period EMA, and 9-period signal line. ? found this online
    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean() #calculates the 12 period Exponential Moving Average (EMA) of close
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean() #same but for 26 period
    df_copy['MACD'] = exp1 - exp2 #calculates the MACD line: 26ema- 12ema, and assigns to new col 'MACD'
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean() #signal line: 9-period EMA of the MACD line itself
    df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal'] #dif between MACD and signal

    df_copy.dropna(inplace=True) #lots of NaN at beginning  of indicator calcs before suff data

    return df_copy #new data frame with techn indicators (and no NaNs)
    
selectedticker_hist_data = historical_prices_data[selected_ticker].copy() 

    # Using function above
se_hist_data_with_indicators = calculate_technical_indicators_manual(selectedticker_hist_data)

    # Print the tail (last few rows) to see the new indicator columns
print("{selected_ticker} Historical Data with Technical Indicators (last 5 rows):")#header (not data)
print(se_hist_data_with_indicators.tail())

def generate_technical_forecast_prompt(selected_ticker: str, df_with_indicators: pd.DataFrame) -> str: #taking stock ticker as a string, and dataframe (should include tech inds), will return string (for ai readbility)

    recent_data = df_with_indicators.tail(10) #making short term forcast so using last 10 lines of df only

    # Formatting for prompt
    data_str = recent_data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']].to_string()

    prompt = f"""
Analyze the following recent technical indicator data for {selected_ticker}: 

{data_str}

please use the data retrieved in calculate_technical_indicators_manual(selectedticker_hist_data)and help create a discounted cash flow analysis for the company in text format with a final stock price obtained by balancing the equity value with enterprise value and number of shares outstanding
express your output as a table with the same lines as in dcf please demonstrate the math being used to calculate the terminal value
Based on this data, provide a concise technical analysis forecast.

Focus on:A
1.  **Current Trend:** What do the SMAs suggest about the short-term and long-term trend?
2.  **Momentum:** What does the RSI indicate (overbought/oversold, bullish/bearish divergence)? What about the MACD (crossovers, histogram)?
3.  **Potential Price Action:** Based on these indicators, what is the most likely immediate future price movement (e.g., bullish, bearish, consolidating)?
4.  **Key Levels:** Are there any implied support or resistance levels?

Be specific and use the indicator values to support your analysis.
"""
    return prompt #PROMPT NEEDS TO BE ADJUSTED FOR OUR NEEDS THIS IS A SAMPLE

ai_tech_prompt = generate_technical_forecast_prompt(selected_ticker, se_hist_data_with_indicators)

print("\nAI prompt for Technical Analysis")
print(ai_tech_prompt)




#
#
#              dcf code
#
#
#




def calculate_fundamental_indicators(financials):
    # Extract individual statements
    income_df = financials.get('income_statement', pd.DataFrame()).copy()
    balance_df = financials.get('balance_sheet', pd.DataFrame()).copy()
    cashflow_df = financials.get('cash_flow', pd.DataFrame()).copy()

    # Move index to column for merging
    for df in [income_df, balance_df, cashflow_df]:
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)

    # Merge on 'Date' across statements
    df = income_df.merge(balance_df, on='Date', how='outer').merge(cashflow_df, on='Date', how='outer')

    # Avoid divide-by-zero errors
    df.replace({0: np.nan}, inplace=True)

    # Normalize column names (snake_case for consistent access)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)

    # print("Columns after merge and normalization:")
    # print(df.columns.tolist())

    # print("\nSample data (first 3 rows):")
    # print(df[['Date'] + [col for col in df.columns if 'Income' in col or 'Revenue' in col or 'Assets' in col or 'Equity' in col]].head(3))


    # === Profitability Ratios ===
    df['Net_Profit_Margin'] = df['Net_Income_Common_Stockholders'] / df['Total_Revenue']
    df['Operating_Margin'] = df['Operating_Income'] / df['Total_Revenue']
    df['Gross_Margin'] = df['Gross_Profit'] / df['Total_Revenue']

    # === Return Ratios ===
    df['ROA'] = df['Net_Income_Common_Stockholders'] / df['Total_Assets']
    df['ROE'] = df['Net_Income_Common_Stockholders'] / df['Stockholders_Equity']

    # === Liquidity Ratios ===
    df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']
    df['Quick_Ratio'] = (df['Current_Assets'] - df['Inventory']) / df['Current_Liabilities']

    # === Leverage Ratios ===
    df['Debt_to_Equity'] = df['Total_Debt'] / df['Stockholders_Equity']
    df['Debt_to_Assets'] = df['Total_Debt'] / df['Total_Assets']

    # === Cash Flow Ratios ===
    df['Operating_Cash_Flow_Ratio'] = df['Operating_Cash_Flow'] / df['Current_Liabilities']
    df['Free_Cash_Flow'] = df['Operating_Cash_Flow'] - df['Capital_Expenditure']

    # df.dropna(inplace=True)

    return df

def output_cleanup(reply) ->str:
    sample_df = pd.read_excel("/Users/kennethle/Downloads/NUS Summer Workshop/Project/DCF_Model_Sample.xlsx")
    sample_as_text = sample_df.to_csv(index=False)
    prompt = rf"""
    Please reorganize {reply} to match with the format of {sample_as_text} but NOT the company. Ensure it is not JSON
    Ensure that the years are columns and the values are rows

    Ensure that the columns will be formatted as separate cells

    example:
            year1  year2  year3  year4   year5
    NOPAT
    D&A
    CapEX
    change in Working Capital
    FCF
    terminal
    share price

    """
    return prompt

def generate_fundamental_dcf_prompt(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
    sample_df = pd.read_excel("/Users/kennethle/Downloads/NUS Summer Workshop/Project/DCF_Model_Sample.xlsx")
    sample_as_text = sample_df.to_csv(index=False)
   
    recent_data = df_with_fundamentals.tail(4)  # Use last 4 periods (e.g., quarters or years)

    # Format relevant rows to string for visibility in the prompt
    data_cols = [
         'Tax_Effect_Of_Unusual_Items', 'Tax_Rate_For_Calcs', 'Normalized_EBITDA', 'Total_Unusual_Items',
    'Total_Unusual_Items_Excluding_Goodwill', 'Net_Income_From_Continuing_Operation_Net_Minority_Interest',
    'Reconciled_Depreciation', 'Reconciled_Cost_Of_Revenue', 'EBITDA', 'EBIT', 'Net_Interest_Income',
    'Interest_Expense', 'Interest_Income', 'Normalized_Income', 'Net_Income_From_Continuing_And_Discontinued_Operation',
    'Total_Expenses', 'Rent_Expense_Supplemental', 'Total_Operating_Income_As_Reported',
    'Diluted_Average_Shares', 'Basic_Average_Shares', 'Diluted_EPS', 'Basic_EPS',
    'Diluted_NI_Availto_Com_Stockholders', 'Net_Income_Common_Stockholders', 'Net_Income',
    'Net_Income_Including_Noncontrolling_Interests', 'Net_Income_Continuous_Operations',
    'Tax_Provision', 'Pretax_Income', 'Other_Income_Expense', 'Other_Non_Operating_Income_Expenses',
    'Special_Income_Charges', 'Gain_On_Sale_Of_Ppe', 'Gain_On_Sale_Of_Business', 'Write_Off',
    'Impairment_Of_Capital_Assets', 'Restructuring_And_Mergern_Acquisition', 'Earnings_From_Equity_Interest',
    'Gain_On_Sale_Of_Security', 'Net_Non_Operating_Interest_Income_Expense', 'Interest_Expense_Non_Operating',
    'Interest_Income_Non_Operating', 'Operating_Income', 'Operating_Expense',
    'Depreciation_Amortization_Depletion_Income_Statement',
    'Depreciation_And_Amortization_In_Income_Statement', 'Selling_General_And_Administration',
    'Gross_Profit', 'Cost_Of_Revenue', 'Total_Revenue', 'Operating_Revenue', 'Treasury_Shares_Number', 'Ordinary_Shares_Number', 'Share_Issued', 'Net_Debt', 'Total_Debt',
    'Tangible_Book_Value', 'Invested_Capital', 'Working_Capital', 'Net_Tangible_Assets',
    'Capital_Lease_Obligations', 'Common_Stock_Equity', 'Total_Capitalization',
    'Total_Equity_Gross_Minority_Interest', 'Stockholders_Equity',
    'Gains_Losses_Not_Affecting_Retained_Earnings', 'Other_Equity_Adjustments', 'Treasury_Stock',
    'Retained_Earnings', 'Additional_Paid_In_Capital', 'Capital_Stock', 'Common_Stock',
    'Preferred_Stock', 'Total_Liabilities_Net_Minority_Interest',
    'Total_Non_Current_Liabilities_Net_Minority_Interest', 'Other_Non_Current_Liabilities',
    'Tradeand_Other_Payables_Non_Current', 'Non_Current_Deferred_Liabilities',
    'Non_Current_Deferred_Revenue', 'Non_Current_Deferred_Taxes_Liabilities',
    'Long_Term_Debt_And_Capital_Lease_Obligation', 'Long_Term_Capital_Lease_Obligation',
    'Long_Term_Debt', 'Current_Liabilities', 'Current_Debt_And_Capital_Lease_Obligation',
    'Current_Capital_Lease_Obligation', 'Current_Debt', 'Other_Current_Borrowings',
    'Payables_And_Accrued_Expenses', 'Current_Accrued_Expenses', 'Interest_Payable', 'Payables',
    'Total_Tax_Payable', 'Income_Tax_Payable', 'Accounts_Payable', 'Total_Assets',
    'Total_Non_Current_Assets', 'Other_Non_Current_Assets', 'Investments_And_Advances',
    'Long_Term_Equity_Investment', 'Investmentsin_Associatesat_Cost',
    'Goodwill_And_Other_Intangible_Assets', 'Goodwill', 'Net_PPE', 'Accumulated_Depreciation',
    'Gross_PPE', 'Other_Properties', 'Machinery_Furniture_Equipment',
    'Buildings_And_Improvements', 'Land_And_Improvements', 'Properties', 'Current_Assets',
    'Other_Current_Assets', 'Prepaid_Assets', 'Inventory', 'Receivables', 'Accounts_Receivable',
    'Cash_Cash_Equivalents_And_Short_Term_Investments', 'Cash_And_Cash_Equivalents', 'Repurchase_Of_Capital_Stock', 'Repayment_Of_Debt', 'Issuance_Of_Debt',
    'Capital_Expenditure', 'Interest_Paid_Supplemental_Data', 'Income_Tax_Paid_Supplemental_Data',
    'End_Cash_Position', 'Beginning_Cash_Position', 'Effect_Of_Exchange_Rate_Changes',
    'Changes_In_Cash', 'Financing_Cash_Flow', 'Cash_Flow_From_Continuing_Financing_Activities',
    'Net_Other_Financing_Charges', 'Proceeds_From_Stock_Option_Exercised', 'Cash_Dividends_Paid',
    'Common_Stock_Dividend_Paid', 'Net_Common_Stock_Issuance', 'Common_Stock_Payments',
    'Net_Issuance_Payments_Of_Debt', 'Net_Short_Term_Debt_Issuance', 'Net_Long_Term_Debt_Issuance',
    'Long_Term_Debt_Payments', 'Long_Term_Debt_Issuance', 'Investing_Cash_Flow',
    'Cash_Flow_From_Continuing_Investing_Activities', 'Net_Other_Investing_Changes',
    'Net_Business_Purchase_And_Sale', 'Sale_Of_Business', 'Purchase_Of_Business',
    'Net_PPE_Purchase_And_Sale', 'Sale_Of_PPE', 'Capital_Expenditure_Reported',
    'Operating_Cash_Flow', 'Cash_Flow_From_Continuing_Operating_Activities',
    'Change_In_Working_Capital', 'Change_In_Other_Working_Capital',
    'Change_In_Payables_And_Accrued_Expense', 'Change_In_Accrued_Expense', 'Change_In_Payable',
    'Change_In_Account_Payable', 'Change_In_Tax_Payable', 'Change_In_Income_Tax_Payable',
    'Change_In_Inventory', 'Change_In_Receivables', 'Changes_In_Account_Receivables',
    'Other_Non_Cash_Items', 'Stock_Based_Compensation', 'Deferred_Tax', 'Deferred_Income_Tax',
    'Depreciation_Amortization_Depletion', 'Depreciation_And_Amortization',
    'Operating_Gains_Losses', 'Gain_Loss_On_Sale_Of_Business', 'Net_Income_From_Continuing_Operations' 
    ]
    fundamental_str = recent_data[data_cols].to_string(index=False)

    prompt = rf"""
    You are a financial analyst tasked with building a full **Discounted Cash Flow (DCF)** model based solely on the following raw fundamental data:
    {data_cols}
    {fundamental_str}


    Please use {sample_as_text} as a reference
    ---
    Your job:
    
    1. **Thoroughly parse the data** and **extract all usable values**, even if the field names are long or inconsistently capitalized. Search **exhaustively** through all available variables. Do not stop early — be methodical and persistent.

    2. Build a complete DCF model including:

    - Historical EBIT, tax rate, D&A, CapEx, and change in working capital
    - Forecast Free Cash Flows (FCF) for 5 years
    - Terminal Value using both:
        - Gordon Growth Model (assume 2.5% long-term growth)
        - Exit Multiple (e.g., EBITDA multiple = 10x)

    3. Calculate the **Weighted Average Cost of Capital (WACC)** using a reasonable assumed capital structure.

    4. **Discount** all FCFs and Terminal Value to Present Value.

    5. Compute:
    - Enterprise Value (EV)
    - Equity Value = EV - Net Debt (Net Debt = Total Debt - Cash & Equivalents)
    - Intrinsic Share Price = Equity Value / Ordinary Shares Outstanding

    6. Present the final DCF as a **clearly structured table in Python `DataFrame` format** with years as columns and the following rows:
    - Projected FCF
    - Present Value of FCF
    - Cumulative PV
    - Terminal Value
    - Enterprise Value
    - Equity Value
    - Implied Share Price

    ---

    **Important**:
    - Do not summarize the data.
    - Do not explain the steps in prose.
    - Just return the final DCF **as a stringified `DataFrame`** that can be parsed in Python **DO NOT WRITE CODE, SIMPLY OUTPUT THE FINALS NUMBERS**.
    - If any key inputs are missing, **assume reasonable values** and clearly indicate them in comments or footnotes within the output.

    Be thorough. The `fundamental_str` data is long — **do not give up early**. Parse all usable values intelligently.
"""



    # Prompt construction
#     prompt = rf"""

# {data_cols}
# {fundamental_str}
# Perform a Discounted Cash Flow (DCF) analysis for {ticker} using the following recent fundamental financial data:

# Please:

# 1. **Project Free Cash Flows (FCF)** for the next 5 years based on historical:
#    \[
#    \\text{{EBIT}} - \\text{{Tax}} - \\text{{Capital\\_Expenditure}} + \\text{{Depreciation\\_and\\_Amortization}} - \\text{{Change\\_in\\_Net\\_Working\\_Capital}}
#    \]

# 2. **Calculate the Terminal Value** using both:
#    - **Gordon Growth Model** (e.g., long-term growth = 2.5%)
#    - **Exit Multiple Method** (e.g., EBITDA multiple)

#    Also, calculate the **Weighted Average Cost of Capital (WACC)**. Show formulas and intermediate steps.

# 3. **Discount all cash flows** (5-year FCFs + Terminal Value) to present value.

# 4. **Compute Enterprise Value (EV)**, then determine **Equity Value**:
#    \[
#    \\text{{Equity\_Value}} = \\text{{Enterprise\_Value}} - \\text{{Net\_Debt}}
#    \]
#    Where:
#    - Net Debt = Total Debt - Cash and Cash Equivalents

# 5. **Determine the intrinsic stock price** by dividing Equity Value by the number of shares:
#    \[
#    \\text{{Intrinsic\_Price}} = \\frac{{\\text{{Equity\_Value}}}}{{\\text{{Ordinary\_Shares\_Number}}}}
#    \]

#    Use the value from the `Ordinary_Shares_Number` field in the balance sheet. If missing, ask the user to provide it.

# 6. **Summarize your DCF in a table**, including:
#    - Year
#    - Projected FCF
#    - Present Value of FCF
#    - Cumulative PV
#    - Terminal Value
#    - Enterprise Value
#    - Equity Value
#    - Implied Share Price

# Then:

# - Provide a concise **fundamental analysis** using these ratios:
#   - Net Profit Margin
#   - Return on Equity (ROE), Return on Assets (ROA)
#   - Current Ratio, Quick Ratio
#   - Debt-to-Equity

# Finally:

# - Offer a valuation: Is the stock **undervalued**, **overvalued**, or **fairly valued** relative to its intrinsic value?

# Use **all numeric values from the table** to support your answer.
# ouput as a string to be converted using
# 1. one with the projections
# 2. with the DCF and valuation calculations

# """



    return prompt



if 'MCD' in financial_statements_data:
    # Extract the raw financials for MCD
    mcd_financials_raw = financial_statements_data['MCD']

    # Copy and fix the structure so the index becomes a 'Date' column
    income_df = mcd_financials_raw['Income Statement'].copy()
    balance_df = mcd_financials_raw['Balance Sheet'].copy()
    cashflow_df = mcd_financials_raw['Cash Flow'].copy()

    for df in [income_df, balance_df, cashflow_df]:
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)

    # Reassemble the corrected input structure using lowercase keys
    mcd_financials_fixed = {
        'income_statement': income_df,
        'balance_sheet': balance_df,
        'cash_flow': cashflow_df
    }

    # Now safely pass into the calculation function
    mcd_fundamental_data = calculate_fundamental_indicators(mcd_financials_fixed)

    if not mcd_fundamental_data.empty:
        ai_dcf_prompt = generate_fundamental_dcf_prompt('MCD', mcd_fundamental_data)

        print("\nAI prompt for Fundamental DCF Analysis ---")
        print(ai_dcf_prompt)
    else:
        print("Cannot generate DCF analysis prompt: MCD fundamental data is empty after processing.")
else:
    print("Cannot generate DCF analysis prompt: MCD not found in financial statements data.")

import requests

def call_ollama(prompt, model='gemma3'):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': prompt,
            'stream': False  # Set to True if you want streamed output
        }
    )
    return response.json()['response']

# Example usage
reply = call_ollama(ai_dcf_prompt)
cleaned = call_ollama(output_cleanup(reply))
company_df = pd.read_fwf(StringIO(cleaned))
print(type(company_df))  # This will tell you what it actually is
company_df.to_excel('output2.xlsx', index=False, engine='openpyxl')

print(reply)

print("program finished")