import pandas as pd
import yfinance as yf
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