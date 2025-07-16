from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import re #used for pattern matching when extracting tickers
from collections import Counter
from fastapi.responses import JSONResponse

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

def call_ollama(prompt, model, repeat=5):
    results = []
    for _ in range(repeat):
        res = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False}
        )
        results.append(res.json()['response']) #converting HTTP body back to a python dict.
    return results

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
    ticker = data.get("ticker", "AAPL")
    stock = yf.Ticker(ticker)
    
    try:
        income = stock.financials.T
        balance = stock.balance_sheet.T
        cashflow = stock.cashflow.T

        financials = {
            'income_statement': income,
            'balance_sheet': balance,
            'cash_flow': cashflow
        }

        fundamentals = calculate_fundamentals(financials)
        fundamentals = fundamentals.tail(4)

        ebit = fundamentals['EBIT'].iloc[-1]
        d_a = fundamentals['Depreciation_And_Amortization'].iloc[-1]
        capex = fundamentals['Capital_Expenditure'].iloc[-1]
        wc = fundamentals['Change_In_Working_Capital'].iloc[-1]
        tax_rate = fundamentals['Tax_Provision'].iloc[-1] / fundamentals['Pretax_Income'].iloc[-1]
        net_debt = fundamentals['Net_Debt'].iloc[-1]

        info = stock.info
        shares_outstanding = info.get('sharesOutstanding', 1)
        wacc = 0.085
        growth_rate = 0.025

        years = list(range(2025, 2030))
        base_fcf = ebit * (1 - tax_rate) + d_a - capex - wc
        fcf_values = [base_fcf * (1 + growth_rate) ** i for i in range(len(years))]

        discounted_fcf = discount(fcf_values, wacc)
        cumulative_pv = np.cumsum(discounted_fcf)

        terminal_value = fcf_values[-1] * (1 + growth_rate) / (wacc - growth_rate)
        discounted_terminal = terminal_value / (1 + wacc) ** len(years)
        enterprise_value = cumulative_pv[-1] + discounted_terminal
        equity_value = enterprise_value - net_debt
        implied_price = equity_value / shares_outstanding

        

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
                   f"the implied price for {ticker.upper()} is approximately ${round(implied_price, 2)}.", 
                  
    "error": None
}


    except Exception as e:
        return {"error": str(e)}
    
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