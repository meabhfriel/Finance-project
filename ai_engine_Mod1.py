import requests #I'll need this to make HTTP calls later, to call ollama
import pandas as pd
import re #used for pattern matching when extracting tickers
from collections import Counter
import yfinance as yfin


# Get Tech Sector Companies from S&P 500

tickers = [
    'ACN', 'ADBE', 'AMD', 'AKAM', 'APH', 'ADI', 'ANSS', 'AAPL', 'AMAT', 'ANET',
    'ADSK', 'AVGO', 'CDNS', 'CDW', 'CSCO', 'CTSH', 'GLW', 'CRWD', 'DDOG', 'DELL',
    'ENPH', 'EPAM', 'FFIV', 'FICO', 'FSLR', 'FTNT', 'IT', 'GEN', 'GDDY', 'HPE',
    'HPQ', 'IBM', 'INTC', 'INTU', 'JBL', 'KEYS', 'KLAC', 'LRCX', 'MCHP', 'MU',
    'MSFT', 'MPWR', 'MSI', 'NTAP', 'NVDA', 'NXPI', 'ON', 'ORCL', 'PLTR', 'PANW',
    'PTC', 'QCOM', 'ROP', 'CRM', 'STX', 'NOW', 'SWKS', 'SMCI', 'SNPS', 'TEL',
    'TDY', 'TER', 'TXN', 'TRMB', 'TYL', 'VRSN', 'WDC', 'WDAY', 'ZBRA'
]

# Pull name + symbol from Yahoo Finance
data = []
for symbol in tickers:
    info = yfin.Ticker(symbol).info
    data.append({
        'Symbol': symbol,
        'Security': info.get('longName', 'Unknown Company')
    })

company_df = pd.DataFrame(data)
companies = company_df.to_string(index=False)

# #Get Energy Sector Companies, can change to any sector or entire S&P 500
# #This is a simple way to get the data, but in production we might want to cache
# #or use a more robust data source.
# url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# sp500 = pd.read_html(url)[0]
# energy_df = sp500[sp500['GICS Sector'] == 'Energy'][['Symbol', 'Security']]
# energy_df.reset_index(drop=True, inplace=True)
# companies = energy_df.to_string(index=False)

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

#Call Ollama
def call_ollama(prompt, model='gemma3', repeat=10):
    results = []
    for _ in range(repeat):
        res = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False}
        )
        results.append(res.json()['response']) #converting HTTP body back to a python dict.
    return results

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

#Run Entire Process and Display Only Final Table
def run(user_input):
    #Find top 3 consistent companies
    initial_prompt = build_prompt(user_input, companies)
    responses = call_ollama(initial_prompt, repeat=5)

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
    final_response = call_ollama(final_prompt, repeat=1)[0]

    # Display only the final table
    print(final_response)     #we only want the user to see the clean concise table

# Run it
run("I want companies that will make me the most money, high volatility")


