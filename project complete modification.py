import pandas as pd
import yfinance as yf
import numpy as np
from io import StringIO
import requests
selected_ticker = 'AAPL' # <--- CHANGE THIS to the stock ticker you want to analyze 
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


#     print(f"Successfully fetched data for {selected_ticker}")

except Exception as e: #assigns any errors to variable e
    print(f"Could not fetch data for {selected_ticker}: {e}") #prints error for us

sp500_df_for_screener = pd.DataFrame.from_dict(all_companies_data, orient='index') #converts dictionary to pandas data frame (from #1 company info)



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


if selected_ticker in financial_statements_data:
    # Extract the raw financials for MCD
    mcd_financials_raw = financial_statements_data[selected_ticker]

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

#     # Now safely pass into the calculation function
    fundamental_data = calculate_fundamental_indicators(mcd_financials_fixed)

#     if not fundamental_data.empty:
#         ai_dcf_prompt = generate_fundamental_dcf_prompt({selected_ticker}, fundamental_data)

#         print("\nGenerated AI prompt for Fundamental DCF Analysis ---")
#         # print(ai_dcf_prompt)
#     else:
#         print("Cannot generate DCF analysis prompt: MCD fundamental data is empty after processing.")
# else:
#     print("Cannot generate DCF analysis prompt: MCD not found in financial statements data.")



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

#
#
#
#
#

####################################################################################
###                                                                              ###
###                                                                              ###
###                            Tax Rate Calculation                              ###
###                                                                              ###
###                                                                              ###
####################################################################################


def tax_rate_calc(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
    """
    Calculate the most recent effective tax rate from financial data.

    Args:
        ticker (str): The stock ticker (for labeling or future use).
        df_with_fundamentals (pd.DataFrame): A DataFrame containing at least 'Tax_Provision' and 'Pretax_Income'.

    Returns:
        str: Formatted effective tax rate (or 'N/A' if not computable).
    """
    # Use the last 2 rows (e.g., most recent quarters or years)
    recent_data = df_with_fundamentals.tail(4)

    # Columns needed
    data_cols_tax_rate = ['Tax_Provision', 'Pretax_Income']

    # Extract the most recent values (last row)
    most_recent = recent_data[data_cols_tax_rate].iloc[0]

    most_recent_tax_provision = most_recent['Tax_Provision']
    most_recent_pretax_income = most_recent['Pretax_Income']

    # Avoid division by zero
    if most_recent_pretax_income != 0:
        effective_tax_rate = most_recent_tax_provision / most_recent_pretax_income
        return effective_tax_rate
    else:
        return 0.2

print("Tax Rate:")
tax_rate = tax_rate_calc(selected_ticker,fundamental_data)
print(tax_rate)

####################################################################################
###                                                                              ###
###                                                                              ###
###                              Net Debt Calculation                            ###
###                                                                              ###
###                                                                              ###
####################################################################################

def net_debt_calc(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
    """
    Calculate the most recent effective tax rate from financial data.

    Args:
        ticker (str): The stock ticker (for labeling or future use).
        df_with_fundamentals (pd.DataFrame): A DataFrame containing at least 'Tax_Provision' and 'Pretax_Income'.

    Returns:
        str: Formatted effective tax rate (or 'N/A' if not computable).
    """
    # Use the last 2 rows (e.g., most recent quarters or years)
    recent_data = df_with_fundamentals.tail(4)

    # Columns needed
    data_cols_net_debt = ['Net_Debt']

    # Extract the most recent values (last row)
    most_recent_nd = recent_data[data_cols_net_debt].iloc[0]

    most_recent_net_debt = most_recent_nd['Net_Debt']

    return most_recent_net_debt


print("Net Debt:")
net_debt = net_debt_calc(selected_ticker,fundamental_data)
print(net_debt)


# def net_debt_calc(ticker: str, df_with_fundamentals: pd.DataFrame) ->str:
#     recent_data = df_with_fundamentals.tail(2)  # Use last 4 periods (e.g., quarters or years)

#     data_cols_net_debt = [
#     'Net_Debt']
    
#     net_data = recent_data[data_cols_net_debt].to_string(index=False)

#     prompt = rf"""
#     You are a financial analyst. Your task is to identify the current net_debt {ticker} using the recent data provided below.

#     Use the following financial information for your analysis:
#     {net_data}

#     Please proceed with the following steps:

#     1. Identify the most recent value for net debt
#     2. **Output the final Net Debt** value as a complete number in standard notation with NO OTHER EXPLANATION OR WORDS, NUMBER ONLY

#     output the final number as an integer
#     **double check to ensure it is only an integer**
#     """

#     return prompt


# print("Locating Net Debt:")
# net_debt_extract = call_ollama(net_debt_calc(selected_ticker,fundamental_data))
# print("Net Debt: ")
# print(net_debt_extract)

# net_debt = float(net_debt_extract)

####################################################################################
###                                                                              ###
###                                                                              ###
###                                 WACC Calculation                             ###
###                                                                              ###
###                                                                              ###
####################################################################################

def wacc_calc(ticker: str, df_with_fundamentals: pd.DataFrame) ->str:
    recent_data = df_with_fundamentals.tail(3)  # Use last 4 periods (e.g., quarters or years)

    data_cols_wacc = [
    # Shares Outstanding
    'Ordinary_Shares_Number',
    'Share_Issued',
    'Diluted_Average_Shares',
    'Basic_Average_Shares',

    # Debt - Short-Term and Long-Term
    'Current_Debt',
    'Other_Current_Borrowings',
    'Current_Debt_And_Capital_Lease_Obligation',
    'Long_Term_Debt',
    'Long_Term_Debt_And_Capital_Lease_Obligation',
    'Total_Debt',

    # Interest Expense (for estimating average debt rate)
    'Interest_Expense',

    # Tax data (for effective tax rate)
    # 'Tax_Provision',
    # 'Pretax_Income',
    # 'Interest_Paid_Supplemental_Data'
    ]

    stock = yf.Ticker(ticker)
    info = stock.info  # Contains metadata and financial metrics
    
    wacc_data_2 = {
        'current_stock_price': info.get('currentPrice'),
        'shares_outstanding': info.get('sharesOutstanding'),
        'beta': info.get('beta'),
        'market_cap': info.get('marketCap'),
        'pe_ratio': info.get('trailingPE'),
        'eps': info.get('trailingEps'),
        'dividend_yield': info.get('dividendYield'),
        'sector': info.get('sector'),
        'long_name': info.get('longName'),
        'exchange': info.get('exchange'),
    }

    
    wacc_data_1 = recent_data[data_cols_wacc].to_string(index=False)

    prompt = rf"""
    You are a financial analyst. Your task is to calculate the Weighted Average Cost of Capital (WACC) for {ticker} using the recent data provided below.

    Use the following financial information for your analysis:
    {wacc_data_1}
    {wacc_data_2}
    {tax_rate}

    Please proceed with the following steps:

    1. **Cost of Equity (Re)**:
    Use the **Capital Asset Pricing Model (CAPM)**:
    \[
    R_e = R_f + \beta \times (R_m - R_f)
    \]
    - Use the provided **beta** value.
    - Assume a **market risk premium (R_m - R_f)** of 5.5%.
    - Use a **risk-free rate (R_f)** of 4.2% (based on current 10-Year U.S. Treasury yield).

    2. **Cost of Debt (Rd)**:
    Estimate the average interest rate on debt using:
    \[
    R_d = \frac{{\text{{Interest_Expense}}}}{{\text{{Average Total Debt}}}}
    \]
    - Use the most recent values for **Interest_Expense** and **Total_Debt** (or the best proxy).
    - If **Interest_Paid_Supplemental_Data** is available, include it in your estimate.
    - Assume tax shield: After-tax cost of debt = \( R_d \times (1 - \text{{Effective Tax Rate}}) \)
    - Effective Tax Rate = {tax_rate}

    3. **Capital Structure**:
    - Estimate the weights of debt and equity:
        \[
        W_e = \frac{{\text{{Market Cap}}}}{{\text{{Market Cap}} + \text{{Total Debt}}}}, \quad
        W_d = \frac{{\text{{Total Debt}}}}{{\text{{Market Cap}} + \text{{Total Debt}}}}
        \]
    - Use **Market Cap** from yfinance data and **Total_Debt** from financials.

    4. **WACC Calculation**:
    \[
    WACC = W_e \times R_e + W_d \times R_d \times (1 - {tax_rate})
    \]

    5. **Output the final WACC** value as a percentage rounded to two decimals. Show all steps, intermediate results, and final formula used.

    If any required value is missing, explain how you'd handle it or what you'd assume.
    """

    return prompt

def wacc_exact(reply) ->str:
    
    prompt = rf"""
    extract only the final wacc number from {reply} expressed **in decimal form NOT percentage form**, **ensure no extraneous spaces or other words**  double check for fromatting

    should be in the same format as:

    0.00
    0.1
    0.2
    1.10
    """

    return prompt
print("Calculating WACC:")
wacc_calculation = call_ollama(wacc_calc(selected_ticker,fundamental_data))
# print(wacc_calculation)
print("Wacc: ")
print(call_ollama(wacc_exact(wacc_calculation)))

####################################################################################
###                                                                              ###
###                                                                              ###
###                                CapEx Calculation                             ###
###                                                                              ###
###                                                                              ###
####################################################################################

def capex_calc(ticker: str, df_with_fundamentals: pd.DataFrame) ->str:
    recent_data = df_with_fundamentals.tail(3)  # Use last 4 periods (e.g., quarters or years)

    data_cols_capex = [
    'Capital_Expenditure']
    
    capex_data = recent_data[data_cols_capex].to_string(index=False)

    prompt = rf"""
    You are a financial analyst. Your task is to identify the current capex {ticker} using the recent data provided below.

    Use the following financial information for your analysis:
    {capex_data}

    Please proceed with the following steps:

    1. Identify the most recent value for capex 
    2. **Output the final Capex** value as a complete number in standard notation with no explanation

    output the final number as an integer
    **double check to ensure it is only an integer**
    """

    return prompt


print("Locating capex:")
capex_extract = call_ollama(capex_calc(selected_ticker,fundamental_data))
print("Capex: ")
print(capex_extract)

capex = float(capex_extract)


####################################################################################
###                                                                              ###
###                                                                              ###
###                                 D&A Calculation                              ###
###                                                                              ###
###                                                                              ###
####################################################################################

def da_calc(ticker: str, df_with_fundamentals: pd.DataFrame) ->str:
    recent_data = df_with_fundamentals.tail(3)  # Use last 4 periods (e.g., quarters or years)

    data_cols_da = [
    'Depreciation_And_Amortization']
    
    da_data = recent_data[data_cols_da].to_string(index=False)

    prompt = rf"""
    You are a financial analyst. Your task is to identify the current depreciation and amortization {ticker} using the recent data provided below.

    Use the following financial information for your analysis:
    {da_data}

    Please proceed with the following steps:

    1. Identify the most recent value for depreciation and amortization
    2. **Output the final Depreciation and Amortization** value as a complete number in standard notation with no explanation

    output the final number as an integer
    **double check to ensure it is only an integer**
    """

    return prompt


print("Locating D&A:")
da_extract = call_ollama(da_calc(selected_ticker,fundamental_data))
print("D&A: ")
print(da_extract)

d_a = float(da_extract)

####################################################################################
###                                                                              ###
###                                                                              ###
###                                EBIT Calculation                              ###
###                                                                              ###
###                                                                              ###
####################################################################################

def ebit_calc(ticker: str, df_with_fundamentals: pd.DataFrame) ->str:
    recent_data = df_with_fundamentals.tail(4)  # Use last 4 periods (e.g., quarters or years)

    data_cols_ebit = [
    'EBIT',
    'Operating_Income'
    ]
    
    ebit_data = recent_data[data_cols_ebit].to_string(index=False)

    prompt = rf"""
    You are a financial analyst. Your task is to identify the current EBIT {ticker} using the recent data provided below.
    Use EBIT, if empty, use Operating_Income


    Use the following financial information for your analysis:
    {ebit_data}

    Please proceed with the following steps:

    1. Identify the most recent value for EBIT (generally the last value)
    2. **Output the final EBIT** value as a complete number in standard notation with no explanation

    output the final number as an integer
    **double check to ensure it is only an integer**
    """

    return prompt


print("Locating EBIT:")
ebit_extract = call_ollama(ebit_calc(selected_ticker,fundamental_data))
print("EBIT: ")
print(ebit_extract)

ebit_0 = float(ebit_extract)

####################################################################################
###                                                                              ###
###                                                                              ###
###                                 WC Calculation                               ###
###                                                                              ###
###                                                                              ###
####################################################################################

def wc_calc(ticker: str, df_with_fundamentals: pd.DataFrame) ->str:
    recent_data = df_with_fundamentals.tail(3)  # Use last 4 periods (e.g., quarters or years)

    data_cols_wc = [
        'Change_In_Working_Capital',
        
        # Current Assets
        'Current_Assets',
        'Cash_And_Cash_Equivalents',
        'Cash_Cash_Equivalents_And_Short_Term_Investments',
        'Inventory',
        'Accounts_Receivable',
        'Other_Current_Assets',
        
        # Current Liabilities
        'Current_Liabilities',
        'Current_Debt',

    ]
    
    wc_data = recent_data[data_cols_wc].to_string(index=False)

    prompt = rf"""
    Using the following financial data for the company, calculate the most recent **Change in Net Working Capital (ΔNWC)**.

    {data_cols_wc}

    You are given values for both **current assets** and **current liabilities**, along with their components, for the last four reporting periods. Your goal is to compute the year-over-year change in net working capital, defined as:

    \[
    \Delta \text{{NWC}} = (\text{{Current Assets}} - \text{{Cash and Equivalents}}) - (\text{{Current Liabilities}} - \text{{Short-Term Debt}})
    \]

    Please apply the formula on a period-to-period basis using the most recent two available time periods and extract values from the most appropriate fields listed below:

    **Current Assets-related fields:**
    - `Current_Assets`
    - `Cash_And_Cash_Equivalents`
    - `Cash_Cash_Equivalents_And_Short_Term_Investments`
    - `Inventory`
    - `Accounts_Receivable`
    - `Other_Current_Assets`

    **Current Liabilities-related fields:**
    - `Current_Liabilities`
    - `Current_Debt`


    **Additional Notes:**
    - Exclude **cash and equivalents** from current assets.
    - Exclude **short-term debt** (i.e., `Current_Debt`, `Other_Current_Borrowings`) from current liabilities.
    - If `Change_In_Working_Capital` is directly available and aligns with your calculation, report it alongside your computed value for verification.

    Return your answer in both **numeric form** and show the **intermediate calculation** based on the latest two available periods. Explain any assumptions you make if data is missing or incomplete.
    """


    return prompt

def wc_exact(reply) ->str:
    
    prompt = rf"""
    extract only the final net change in working capital number from {reply} expressed **in numerical form NOT scientific form**, **ensure no extraneous spaces**  double check for fromatting

    should be in the same format as:

    0.00
    0.1
    0.2
    1.10
    """

    return prompt


print("Locating WC:")
wc_extract = call_ollama(wc_calc(selected_ticker,fundamental_data))
print("WC: ")
print(call_ollama(wc_exact(wc_extract)))

wc_change = float(call_ollama(wc_exact(wc_extract)))

# Assumptions

wacc = float(call_ollama(wacc_exact(wacc_calculation)))
stock = yf.Ticker(selected_ticker)
info = stock.info  # Contains metadata and financial metrics

e_0 = info.get('sharesOutstanding')  # Shares outstanding
growth_rate = 0.025  # 2.5%
# years = ["year" "1", "2","3", "4", "5", "6"]

# Discount function
def discount(cash_flows, discount_rate, periods=None):
    cash_flows = np.array(cash_flows, dtype=float)
    if periods is None:
        periods = np.arange(1, len(cash_flows) + 1)
    else:
        periods = np.array(periods)
    return cash_flows / (1 + discount_rate) ** periods

# 1. Calculate FCFs
years = list(range(2025, 2030))

# Set up your DCF DataFrame with years as columns

# Calculate FCF values (same across all years in this example)
fcf_values = [ebit_0 * (1 - tax_rate) - capex - wc_change + d_a] * len(years)

# Discounted FCF
discounted_fcf = discount(fcf_values, wacc)

# Cumulative PV
cumulative_pv = np.cumsum(discounted_fcf)

# Terminal Value (using Gordon Growth Model)
terminal_value = fcf_values[-1] * (1 + growth_rate) / (wacc - growth_rate)
discounted_terminal = terminal_value / (1 + wacc) ** len(years)

enterprise_value = cumulative_pv[-1] + discounted_terminal
equity_value = enterprise_value - net_debt
implied_price = equity_value / e_0

# Create a dictionary of rows
dcf_data = {
    "EBIT": [ebit_0] * len(years),
    "Tax Rate": [tax_rate] * len(years),
    "D&A": [d_a] * len(years),
    "Capex": [capex] * len(years),
    "WC Change": [wc_change] * len(years),
    "FCF": fcf_values,
    "Discounted FCF": discounted_fcf,
    "Cumulative PV": cumulative_pv,
}

# Convert to DataFrame and transpose
# df = pd.DataFrame(dcf_data, index=years).T
# df.columns = [str(year) for year in years]

# Convert to DataFrame and transpose
df = pd.DataFrame(dcf_data, index=years).T

# Add the row names as a column
df.reset_index(inplace=True)
df.rename(columns={'index': 'Metric'}, inplace=True)



# Add final rows separately
summary_df = pd.DataFrame({
    "Metric": ["WACC","Terminal Value (PV)", "Enterprise Value", "Equity Value", "Implied Share Price"],
    "Value": [wacc,terminal_value, enterprise_value, equity_value, implied_price]
})

# Display results
print(df.to_string(index=False))
print("\nSummary:")
print(summary_df.to_string(index=False))

with pd.ExcelWriter('Auto_DCF.xlsx') as writer: 
    df.to_excel(writer, sheet_name='Cash Flow Summary',index=False, engine='openpyxl')
    summary_df.to_excel(writer, sheet_name='DCF',index=False, engine='openpyxl')
