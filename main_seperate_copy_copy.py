from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import re #used for pattern matching when extracting tickers
from collections import Counter
from fastapi.responses import JSONResponse
financial_statements_data = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



### 2. 技术指标计算函数 ###
def calculate_technical_indicators_manual(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()

    delta = df_copy['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']

    df_copy.dropna(inplace=True)
    return df_copy

### 3. 技术指标分析 prompt 构造 ###
def generate_technical_forecast_prompt(ticker: str, df_with_indicators: pd.DataFrame) -> str:
    recent = df_with_indicators.tail(10)
    data_str = recent[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']].to_string()

    prompt = f"""
Analyze the following recent technical indicator data for {ticker}:

{data_str}

Please provide a concise technical analysis forecast including trend, momentum, potential price action, and key levels. Also generate a DCF table with inputs and explain terminal value calculation.
"""
    return prompt

### 4. 通用 LLM 请求方法 ###

def call_ollama(prompt, model, repeat):
    results = []
    for _ in range(repeat):
        res = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False}
        )
        results.append(res.json()['response']) #converting HTTP body back to a python dict.
    return results[0] if repeat == 1 else results

###1.Module 1###
# Get Tech Sector Companies from S&P 500

# tickers = [
#     'ACN', 'ADBE', 'AMD', 'AKAM', 'APH', 'ADI', 'ANSS', 'AAPL', 'AMAT', 'ANET',
#     'ADSK', 'AVGO', 'CDNS', 'CDW', 'CSCO', 'CTSH', 'GLW', 'CRWD', 'DDOG', 'DELL',
#     'ENPH', 'EPAM', 'FFIV', 'FICO', 'FSLR', 'FTNT', 'IT', 'GEN', 'GDDY', 'HPE',
#     'HPQ', 'IBM', 'INTC', 'INTU', 'JBL', 'KEYS', 'KLAC', 'LRCX', 'MCHP', 'MU',
#     'MSFT', 'MPWR', 'MSI', 'NTAP', 'NVDA', 'NXPI', 'ON', 'ORCL', 'PLTR', 'PANW',
#     'PTC', 'QCOM', 'ROP', 'CRM', 'STX', 'NOW', 'SWKS', 'SMCI', 'SNPS', 'TEL',
#     'TDY', 'TER', 'TXN', 'TRMB', 'TYL', 'VRSN', 'WDC', 'WDAY', 'ZBRA'
# ]

# # Pull name + symbol from Yahoo Finance
# data = []
# for symbol in tickers:
#     info = yf.Ticker(symbol).info
#     data.append({
#         'Symbol': symbol,
#         'Security': info.get('longName', 'Unknown Company')
#     })

# company_df = pd.DataFrame(data)
# companies = company_df.to_string(index=False)

#Get Energy Sector Companies, can change to any sector or entire S&P 500
#This is a simple way to get the data, but in production we might want to cache
#or use a more robust data source.
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500 = pd.read_html(url)[0]
energy_df = sp500[sp500['GICS Sector'] == 'Energy'][['Symbol', 'Security']]
energy_df.reset_index(drop=True, inplace=True)
companies = energy_df.to_string(index=False)
company_df = energy_df

#Building Prompt With Few-Shot strategy 
def build_prompt(user_preference: str, companies: str) -> str: #all this is gonna take as an input is user preference and company list
    
    #want it to follow strict procedure 
    #so I have given it an example and clear and concise instructions
    #storing multiline string as variable 'few_shot' to be used later
    few_shot = """
Example:
User request: "low volatility"
|--------|------------------|------------------------------|
| XOM    | Exxon Mobil      | . Low beta                   |
|        |                  | . Stable dividend history    |
|        |                  | . Large market cap           |
|--------|------------------|------------------------------|

Now respond to this new user request:
"""
  #My prompt f string  
    prompt = f"""
You are a financial AI assistant trained specifically to screen energy sector stocks based strictly on user-defined investment preferences.
Do not ask questions. Do not speculate. Use only the data provided. Always return results in a table.
This is not a conversation. Simply return a structured result based on the user input.

{few_shot}
"{user_preference}"

You have access to these energy companies from the S&P 500:
{companies}

Output format requirements:
- Show only the top 3 companies.
- "Rationale" must use bullet points (use `.` for each point)
- Max 3 bullet points per company. 
- Every column must be fixed-width using pipes (`|`), padded so all rows align.
- Do not wrap text or change structure.
"""
    return prompt.strip()

# #Call Ollama
# def call_ollama(prompt, model='gemma3', repeat=10):
#     results = []
#     for _ in range(repeat):
#         res = requests.post(
#             'http://localhost:11434/api/generate',
#             json={'model': model, 'prompt': prompt, 'stream': False}
#         )
#         results.append(res.json()['response']) #converting HTTP body back to a python dict.
#     return results

#Extract Tickers From Each Response
def extract_tickers(text):
    tickers = []
    for line in text.splitlines():
        match = re.match(r"\|\s*([A-Z]{1,5})\s*\|", line)  #using raw string form 
        if match:
            tickers.append(match.group(1)) #grabbing the ticker if it matches the form we want 
    return tickers

#Final Strict Prompt
def build_final_table_prompt(user_input, filtered_companies):
    return f"""
You are a financial AI assistant. Using ONLY the companies listed below, return a final table.
Do not make assumptions. Do not add any other companies.
Only use the information provided.

User request:
"{user_input}"

Companies to use:
{filtered_companies}

Output format:
- Must list all 3 companies, no more, no less.
- Rationale must be exactly 3 bullet points (use `.`).
- Rationale must be only 10 
- Use fixed-width format with aligned pipes.
- Do not include any extra text.
- Do NOT include any text before or after the table


Example format:

| Symbol | Company Name     | Rationale                    |
|--------|------------------|------------------------------|
| XOM    | Exxon Mobil      | . Bullet 1                   |
|        |                  | . Bullet 2                   |
|        |                  | . Bullet 3                   |
|--------|------------------|------------------------------|
""".strip()

def calculate_fundamentals(financials):
    income_df = financials['income_statement'].copy()
    balance_df = financials['balance_sheet'].copy()
    cashflow_df = financials['cash_flow'].copy()

    for df in [income_df, balance_df, cashflow_df]:
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)

    df = income_df.merge(balance_df, on='Date', how='outer').merge(cashflow_df, on='Date', how='outer')
    df.replace({0: np.nan}, inplace=True)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)

    df['Net_Profit_Margin'] = df['Net_Income_Common_Stockholders'] / df['Total_Revenue']
    df['ROE'] = df['Net_Income_Common_Stockholders'] / df['Stockholders_Equity']
    df['Debt_to_Equity'] = df['Total_Debt'] / df['Stockholders_Equity']
    df['Free_Cash_Flow'] = df['Operating_Cash_Flow'] - df['Capital_Expenditure']
    return df


def discount(cash_flows, discount_rate, periods=None):
    cash_flows = np.array(cash_flows, dtype=float)
    if periods is None:
        periods = np.arange(1, len(cash_flows) + 1)
    else:
        periods = np.array(periods)
    return cash_flows / (1 + discount_rate) ** periods

### 5. POST 接口：用户偏好选股分析 ###




@app.post("/calculate_dcf")
async def calculate_dcf(request: Request):
    data = await request.json()
    selected_ticker = data.get("ticker", "AAPL")
    stock = yf.Ticker(selected_ticker)
    financial_statements_data[selected_ticker] = {
        'Income Statement': stock.financials.T, #stock.financials retrieves the annual income statements for the ticker as a Pandas DataFrame. 
        'Balance Sheet': stock.balance_sheet.T, #stock.balance_sheet retrieves the annual balamce sheets
        'Cash Flow': stock.cashflow.T #stock.cashflow retrieves annual cashflow statements
    } #.T to transpose so that  dates are in rows and financial data in columns (to help make data easier to work with), and then we store these 3 financial statemen's info in financial_statements_data[ticker]

    
    
    try:
        def calculate_fundamental_indicators(financials):
            financials_raw = financial_statements_data[selected_ticker]
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

            return df
        
        if selected_ticker in financial_statements_data:
            # Extract the raw financials for MCD
            financials_raw = financial_statements_data[selected_ticker]

            # Copy and fix the structure so the index becomes a 'Date' column
            income_df = financials_raw['Income Statement'].copy()
            balance_df = financials_raw['Balance Sheet'].copy()
            cashflow_df = financials_raw['Cash Flow'].copy()

            for df in [income_df, balance_df, cashflow_df]:
                df['Date'] = df.index
                df.reset_index(drop=True, inplace=True)

            # Reassemble the corrected input structure using lowercase keys
            financials_fixed = {
                'income_statement': income_df,
                'balance_sheet': balance_df,
                'cash_flow': cashflow_df
            }
            # print("First row of Income Statement:")
            # print(income_df.head(1).T)  # Transpose to view vertical

        #     # Now safely pass into the calculation function


            fundamental_data = calculate_fundamental_indicators(financials_fixed)
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

            If WACC is below 5%, default to 10%

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
        wacc_calculation = call_ollama(wacc_calc(selected_ticker,fundamental_data),"gemma3",1)
        # print(wacc_calculation)
        print("Wacc: ")
        print(call_ollama(wacc_exact(wacc_calculation),"gemma3",1))

        ####################################################################################
        ###                                                                              ###
        ###                                                                              ###
        ###                                CapEx Calculation                             ###
        ###                                                                              ###
        ###                                                                              ###
        ####################################################################################

        def capex_forecast_prompt(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
            recent_data = df_with_fundamentals.tail(6)  # use more periods if available

            data_cols_capex = ['Capital_Expenditure']
            capex_data = recent_data[data_cols_capex].to_string(index=False)

            prompt = rf"""
            You are a financial analyst tasked with forecasting **Capital Expenditures (CapEx)** for {ticker} over the next 5 years.

            You are provided with historical values for:
            - Capital_Expenditure

            Here is the historical CapEx data:
            {capex_data}

            Please follow these instructions:

            1. Analyze the historical CapEx trend.
            2. Make a reasonable assumption about CapEx growth or stability. If erratic, consider smoothing or using a moving average.
            3. Forecast CapEx for the next 5 years using your assumption.
            4. Output the projected CapEx values **as a list of 5 numbers**, in plain decimal format, without dollar signs or commas and as full numbers NOT in scientific notation.

            **Format your output like this:**
            CapEx Growth Assumption: ±X% or constant
            Forecast: [value1, value2, value3, value4, value5]
            """

            return prompt.strip()

        print("Generating CapEx forecast:")
        capex_forecast_output = call_ollama(capex_forecast_prompt(selected_ticker, fundamental_data),"gemma3",1)
        print("AI Response:\n", capex_forecast_output)

        import re

        match = re.search(r"Forecast:\s*\[(.*?)\]", capex_forecast_output)
        if match:
            capex = [float(x.strip()) for x in match.group(1).split(',')]
            print("match found")
        else:
            raise ValueError("Could not parse CapEx forecast from AI response")

        # Optional: Extract CapEx growth assumption
        growth_match = re.search(r"CapEx Growth Assumption:\s*([+-]?\d+\.?\d*)%", capex_forecast_output)
        if growth_match:
            capex_growth_rate = float(growth_match.group(1)) / 100
        else:
            capex_growth_rate = None

        print("done with capex")

        ####################################################################################
        ###                                                                              ###
        ###                                                                              ###
        ###                                 D&A Calculation                              ###
        ###                                                                              ###
        ###                                                                              ###
        ####################################################################################

        def da_forecast_prompt(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
            recent_data = df_with_fundamentals.tail(6)  # Use more data if available

            data_cols_da = [
                'Depreciation_And_Amortization'
            ]

            da_data = recent_data[data_cols_da].to_string(index=False)

            prompt = rf"""
            You are a financial analyst tasked with forecasting **Depreciation & Amortization (D&A)** for {ticker} over the next 5 years.

            You are provided with historical data, including:
            - Depreciation_And_Amortization


            Here is the historical D&A data:
            {da_data}

            Please follow these steps:

            1. Analyze the historical trend of D&A.
            2. Make a realistic assumption about how D&A will evolve over the next 5 years (constant, increasing, or decreasing).
            3. Forecast D&A for each of the next 5 years using your assumption.
            4. Output only the projected D&A values **as a list of 5 numbers**, in plain decimal format, and without symbols or commas and as full numbers NOT in scientific notation.

            **Format your answer like this:**
            D&A Growth Assumption: ±X% or constant
            Forecast: [value1, value2, value3, value4, value5]
            """

            return prompt.strip()

        print("Generating D&A forecast:")
        da_forecast_output = call_ollama(da_forecast_prompt(selected_ticker, fundamental_data),"gemma3",1)
        print("AI Response:\n", da_forecast_output)

        match = re.search(r"Forecast:\s*\[(.*?)\]", da_forecast_output)
        if match:
            d_a = [float(x.strip()) for x in match.group(1).split(',')]
        else:
            raise ValueError("Could not parse D&A forecast from AI response")

        # Optional: Extract D&A growth assumption
        growth_match = re.search(r"D&A Growth Assumption:\s*([+-]?\d+\.?\d*)%", da_forecast_output)
        if growth_match:
            da_growth_rate = float(growth_match.group(1)) / 100
        else:
            da_growth_rate = None


        ####################################################################################
        ###                                                                              ###
        ###                                                                              ###
        ###                                EBIT Calculation                              ###
        ###                                                                              ###
        ###                                                                              ###
        ####################################################################################

        def ebit_forecast_prompt(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
            recent_data = df_with_fundamentals.tail(8)  # Use more periods if available

            data_cols_ebit = ['EBIT', 'Operating_Income']
            ebit_data = recent_data[data_cols_ebit].to_string(index=False)

            prompt = rf"""
            You are a financial analyst tasked with forecasting EBIT for the company {ticker}.

            You are provided with recent historical financial data for EBIT and Operating Income. 
            If EBIT is missing, you may use Operating_Income as a proxy.

            Here is the historical data:
            {ebit_data}

            Please perform the following:

            1. Analyze the EBIT trend (use Operating_Income only if EBIT is not available).
            2. Make a reasonable assumption about the **annual EBIT growth rate** (explicitly state your assumption in %).
            3. Forecast EBIT for the next 5 years using that growth rate.
            4. Output the projected EBIT values **as a list of 5 integers**, one per year, in standard notation (no symbols or commas) and as full numbers NOT in scientific notation.

            **Format your response exactly as follows:**
            EBIT Growth Rate: X%
            Forecast: [value1, value2, value3, value4, value5]
            """

            return prompt.strip()

        print("Generating EBIT forecast:")
        ebit_forecast_output = call_ollama(ebit_forecast_prompt(selected_ticker, fundamental_data),"gemma3",1)
        print("AI Response:\n", ebit_forecast_output)

        import re

        match = re.search(r"Forecast:\s*\[(.*?)\]", ebit_forecast_output)
        if match:
            ebit_0 = [float(x.strip()) for x in match.group(1).split(',')]
        else:
            raise ValueError("Could not parse forecast from AI response")

        # Optional: get growth rate too
        growth_match = re.search(r"EBIT Growth Rate:\s*([\d.]+)%", ebit_forecast_output)
        if growth_match:
            ebit_growth_rate = float(growth_match.group(1)) / 100
        else:
            ebit_growth_rate = None


        ####################################################################################
        ###                                                                              ###
        ###                                                                              ###
        ###                                 WC Calculation                               ###
        ###                                                                              ###
        ###                                                                              ###
        ####################################################################################

        def wc_forecast_prompt(ticker: str, df_with_fundamentals: pd.DataFrame) -> str:
            recent_data = df_with_fundamentals.tail(6)  # use more history if available

            data_cols_wc = [
                'Change_In_Working_Capital',
                'Current_Assets',
                'Current_Liabilities'
            ]

            wc_data = recent_data[data_cols_wc].to_string(index=False)

            prompt = rf"""
            You are a financial analyst tasked with forecasting the **Change in Net Working Capital (ΔNWC)** for {ticker} over the next 5 years.

            You are provided with historical data including:
            - Change_In_Working_Capital (if available)
            - Current assets and liabilities and their components

            Here is the historical data:
            {wc_data}

            Please follow these instructions:
            1. Compute ΔNWC over the last 2-3 periods using this formula:
            ΔNWC = (Current Assets - Cash) - (Current Liabilities - Short-Term Debt)
            2. Analyze the trend and assume a **realistic growth rate or stabilization trend** for ΔNWC.
            3. Forecast the ΔNWC for the next 5 years using your assumption.
            4. Output the projected ΔNWC values **as a list of 5 numbers**, in plain decimal format and as full numbers NOT in scientific notation.

            **Format your answer like this:**
            ΔNWC Growth Assumption: ±X% or constant
            Forecast: [value1, value2, value3, value4, value5]
            """

            return prompt.strip()

        print("Generating ΔNWC forecast:")
        wc_forecast_output = call_ollama(wc_forecast_prompt(selected_ticker, fundamental_data),"gemma3",1)
        print("AI Response:\n", wc_forecast_output)

        import re

        match = re.search(r"Forecast:\s*\[(.*?)\]", wc_forecast_output)
        if match:
            wc_change = [float(x.strip()) for x in match.group(1).split(',')]
        else:
            raise ValueError("Could not parse ΔNWC forecast from AI response")

        # Optional: get growth assumption
        growth_match = re.search(r"ΔNWC Growth Assumption:\s*([+-]?\d+\.?\d*)%", wc_forecast_output)
        if growth_match:
            wc_growth_rate = float(growth_match.group(1)) / 100
        else:
            wc_growth_rate = None


        ####################################################################################
        ###                                                                              ###
        ###                                                                              ###
        ###                                DCF Calculation                               ###
        ###                                                                              ###
        ###                                                                              ###
        ####################################################################################

        wacc = float(call_ollama(wacc_exact(wacc_calculation),"gemma3",1))
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


        # Calculate FCF values (same across all years in this example)
        fcf_values = [
            ebit * (1 - tax_rate) - cap - wc + da
            for ebit, cap, wc, da in zip(ebit_0, capex, wc_change, d_a)
        ]

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
            "EBIT": ebit_0,
            "Tax Rate": [tax_rate] * len(years),
            "D&A": d_a,
            "Capex": capex,
            "WC Change": wc_change,
            "FCF": fcf_values,
            "Discounted FCF": discounted_fcf,
            "Cumulative PV": cumulative_pv,
        }

        # Convert to DataFrame and transpose
        df = pd.DataFrame(dcf_data, index=years).T

        # Add the row names as a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Metric'}, inplace=True)



        # Add final rows separately
        summary_df = pd.DataFrame({
            "Metric": ["WACC","Net Debt","Terminal Value (PV)", "Enterprise Value", "Equity Value", "Implied Share Price"],
            "Value": [wacc,net_debt,terminal_value, enterprise_value, equity_value, implied_price]
        })

        # Display results
        print(df.to_string(index=False))
        print("\nSummary:")
        print(summary_df.to_string(index=False))

        with pd.ExcelWriter('Auto_DCF.xlsx') as writer: 
            df.to_excel(writer, sheet_name='Cash Flow Summary',index=False, engine='openpyxl')
            summary_df.to_excel(writer, sheet_name='DCF',index=False, engine='openpyxl')
                        
        return {
        "valuation": {
        "FCF (Year 1)": round(fcf_values[0], 2),
        "Terminal Value": round(terminal_value, 2),
        "Enterprise Value": round(enterprise_value, 2),
        "Equity Value": round(equity_value, 2),
        "Implied Price": round(implied_price, 2),
        "WACC": round(wacc, 4),
        "Tax Rate": round(tax_rate, 4)
        },
        "fcf_values": fcf_values,
        "discounted_fcf": discounted_fcf.tolist(), 
         "explanation": f"Based on a {round(wacc*100,1)}% WACC and {round(growth_rate*100,1)}% perpetual growth, "
                   f"the implied price for {selected_ticker.upper()} is approximately ${round(implied_price, 2)}.", 
                  
    "error": None
}

    except Exception as e:
        return {"error": str(e)}
    


    #
    #
    #
    #
    #
    #
    #
    #
    
@app.post("/analyze")
async def analyze_stock(request: Request):
    data = await request.json()
    user_input= data.get("preference", "I want stable, US-based energy companies with solid growth.")
    model = data.get("model", "gemma3")  # read selected model from frontend
    initial_prompt = build_prompt(user_input, companies)
    responses = call_ollama(initial_prompt, model, repeat=5)

    all_tickers = [t for r in responses for t in extract_tickers(r)] #using double loop For each r (response) in responses
                                                                     #it calls extract_tickers(r) to get the tickers from that table.
    ticker_counts = Counter(all_tickers)
    top_3 = [ticker for ticker, _ in ticker_counts.most_common(3)]    #these next few lines combine previous logic and code 
                                                                      #for a consistency check we only want the tickers that appeared the
                                                                      # most in repeated trials

    #Filter company info to just top 3
    filtered_df = company_df[company_df['Symbol'].isin(top_3)]
    filtered_companies = filtered_df.to_string(index=False)

    # Generate final table
    final_prompt = build_final_table_prompt(user_input, filtered_companies) 
    final_response = call_ollama(final_prompt, model, repeat=1)[0]

    return {"reply": final_response,"top_3": top_3,"Finalprompt": final_prompt,"Final response:\n": final_response}
#DONT CHANGE THE RETURN JSON, IT ACCIDENTALLY TRIGGER SOMETHING

### 拆分功能 ###

cached_data = {}


@app.post("/fetch_data")
async def fetch_data(request: Request):
    data = await request.json()
    ticker = data.get("ticker")
    if not ticker:
        return {"error": "Please provide 'ticker'."}

    try:
        df = yf.download(ticker, period="3mo", interval="1d").reset_index()
        if df.empty:
            return {"error": f"No data found for {ticker}"}

        df = df[['Date', 'Close']]
        df['Date'] = df['Date'].astype(str)
        df.columns = [str(col) for col in df.columns]
        cached_data[ticker] = df.copy()
        return JSONResponse(content={"data": df.to_dict(orient="records")})

    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}

    
def flatten_col_name(col_str):
    if col_str.startswith("('") and col_str.endswith("')"):
        parts = col_str.strip("()").split(",")
        return parts[0].strip(" '\"")
    return col_str


@app.post("/analyze_indicators")
async def analyze_indicators(request: Request):
    data = await request.json()
    ticker = data.get("ticker")
    raw_data = data.get("raw_data")

    print("Received raw_data:", raw_data)  # 打印 raw_data
    if not raw_data or not ticker:
        return {"error": "Must provide both 'ticker' and 'raw_data'"}

 
    df = pd.DataFrame(raw_data)

    
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]


    df.columns = [flatten_col_name(col) for col in df.columns]
    print("Columns after flatten:", df.columns.tolist())
    
    if 'Close' not in df.columns:
        return {"error": "'Close' column not found in data."}
    if 'Date' not in df.columns:
        return {"error": "'Date' column not found in data."}

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")

    # 缓存结果（如果你想保留）
    cached_data[ticker] = df

    return {
        "message": "Indicators calculated successfully.",
        "indicators": df.to_dict(orient="records")
    }

    


# LLM 分析 + 图表接口
@app.post("/chart_data")
async def chart_data(request: Request):
    data = await request.json()
    ticker = data.get("ticker")

    if ticker not in cached_data:
        return {"error": "No data with indicators found."}

    df = cached_data[ticker]
    df_tail = df.tail(50)

    # 生成 prompt
    prompt = f"""
Analyze the following technical indicator data for {ticker}:

{df.tail(10)[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']].to_string(index=False)}

Please provide a concise technical analysis forecast including trend, momentum, potential price action, and key levels. Also generate a DCF table with inputs and explain terminal value calculation.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3", "prompt": prompt, "stream": False},
            timeout=90
        )
        response.raise_for_status()
        reply = response.json().get("response", "No reply")
    except Exception as e:
        reply = f"LLM request failed: {str(e)}"

    plot_data = {
        "labels": df_tail["Date"].tolist(),
        "close": df_tail["Close"].tolist(),
        "sma20": df_tail["SMA_20"].tolist(),
        "sma50": df_tail["SMA_50"].tolist(),
        "rsi": df_tail["RSI"].tolist()
    }

    return {
        "reply": reply,
        "plot_data": plot_data
    }