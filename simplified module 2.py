import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. Data Fetching ---
def fetch_stock_data(ticker: str):
    """
    Fetches comprehensive stock data (info, historical prices, financial statements)
    for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        tuple: (info_dict, hist_df, financial_statements_dict)
               Returns (None, None, None) if data cannot be fetched.
    """
    print(f"Fetching detailed data for {ticker} via yfinance...")
    try:
        stock = yf.Ticker(ticker)

        info = stock.info
        hist = stock.history(period="1y")

        financial_statements = {
            'income_statement': stock.financials.T,
            'balance_sheet': stock.balance_sheet.T,
            'cash_flow': stock.cashflow.T
        }
        print(f"Successfully fetched data for {ticker}")
        return info, hist, financial_statements

    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return None, None, None

# --- 2. Technical Indicators (existing function) ---
def calculate_technical_indicators_manual(df):
    if 'Close' not in df.columns:
        print("DataFrame must contain a 'Close' column for technical analysis.")
        return df.copy()

    df_copy = df.copy()
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()

    window_length = 14
    delta = df_copy['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=window_length - 1, min_periods=window_length).mean()
    avg_loss = loss.ewm(com=window_length - 1, min_periods=window_length).mean()

    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']

    df_copy.dropna(inplace=True)
    return df_copy

# --- 3. Fundamental Indicators & Ratio Calculation ---
def calculate_fundamental_indicators(financials: dict):
    """
    Calculates various financial ratios from raw financial statements.

    Args:
        financials (dict): Dictionary containing 'income_statement', 'balance_sheet', 'cash_flow' DataFrames.

    Returns:
        pd.DataFrame: Merged DataFrame with calculated fundamental ratios.
    """
    income_df = financials.get('income_statement', pd.DataFrame()).copy()
    balance_df = financials.get('balance_sheet', pd.DataFrame()).copy()
    cashflow_df = financials.get('cash_flow', pd.DataFrame()).copy()

    for df in [income_df, balance_df, cashflow_df]:
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)

    df = income_df.merge(balance_df, on='Date', how='outer').merge(cashflow_df, on='Date', how='outer')
    df.replace({0: np.nan}, inplace=True)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)

    # === Profitability Ratios ===
    df['Net_Profit_Margin'] = df.get('Net_Income_Common_Stockholders', np.nan) / df.get('Total_Revenue', np.nan)
    df['Operating_Margin'] = df.get('Operating_Income', np.nan) / df.get('Total_Revenue', np.nan)
    df['Gross_Margin'] = df.get('Gross_Profit', np.nan) / df.get('Total_Revenue', np.nan)

    # === Return Ratios ===
    df['ROA'] = df.get('Net_Income_Common_Stockholders', np.nan) / df.get('Total_Assets', np.nan)
    df['ROE'] = df.get('Net_Income_Common_Stockholders', np.nan) / df.get('Stockholders_Equity', np.nan)

    # === Liquidity Ratios ===
    df['Current_Ratio'] = df.get('Current_Assets', np.nan) / df.get('Current_Liabilities', np.nan)
    df['Quick_Ratio'] = (df.get('Current_Assets', np.nan) - df.get('Inventory', np.nan)) / df.get('Current_Liabilities', np.nan)

    # === Leverage Ratios ===
    df['Debt_to_Equity'] = df.get('Total_Debt', np.nan) / df.get('Stockholders_Equity', np.nan)
    df['Debt_to_Assets'] = df.get('Total_Debt', np.nan) / df.get('Total_Assets', np.nan)

    # === Cash Flow Ratios ===
    df['Operating_Cash_Flow_Ratio'] = df.get('Operating_Cash_Flow', np.nan) / df.get('Current_Liabilities', np.nan)
    df['Free_Cash_Flow'] = df.get('Operating_Cash_Flow', np.nan) - df.get('Capital_Expenditure', np.nan)

    return df

# --- 4. DCF Parameters Calculation (Modified: WACC calculation removed) ---
def dcf_parameters_calculator(ticker_info: dict, fundamental_data_df: pd.DataFrame):
    """
    Calculates or defines key DCF parameters: Terminal Growth Rate, Exit Multiple, Net Debt, Shares Outstanding.
    WACC calculation logic has been removed as per request.

    Args:
        ticker_info (dict): Dictionary from yfinance stock.info.
        fundamental_data_df (pd.DataFrame): Processed DataFrame with fundamental data and ratios.

    Returns:
        dict: A dictionary containing DCF parameters.
    """
    params = {
        'terminal_growth_rate': 0.025,  # A common conservative assumption (2.5%)
        'exit_multiple_value': 10.0,    # Common assumption for EV/EBITDA multiple (10x)
        'calculated_wacc': np.nan,      # WACC explicitly set to NaN (removed calculation logic)
        'calculated_net_debt': np.nan,
        'calculated_shares_outstanding': np.nan,
        'wacc_breakdown': {}            # Will be empty for WACC components
    }

    if not fundamental_data_df.empty:
        latest_data = fundamental_data_df.iloc[-1]

        total_debt = latest_data.get('Total_Debt', np.nan)
        cash_equiv = latest_data.get('Cash_And_Cash_Equivalents', np.nan)

        if not pd.isna(total_debt):
            params['calculated_net_debt'] = total_debt - (cash_equiv if not pd.isna(cash_equiv) else 0)
        elif 'Net_Debt' in latest_data and not pd.isna(latest_data['Net_Debt']):
             params['calculated_net_debt'] = latest_data['Net_Debt']

        params['calculated_shares_outstanding'] = ticker_info.get('sharesOutstanding', np.nan)
        if pd.isna(params['calculated_shares_outstanding']) and 'Ordinary_Shares_Number' in latest_data:
             params['calculated_shares_outstanding'] = latest_data['Ordinary_Shares_Number']

    print("\nDCF Parameter Calculation Summary (Script-Defined)")
    print(f"Terminal Growth Rate: {params['terminal_growth_rate']:.1%}")
    print(f"Exit Multiple (EV/EBITDA): {params['exit_multiple_value']:.1f}x")
    print(f"Calculated Net Debt: ${params['calculated_net_debt']:,.0f}" if not pd.isna(params['calculated_net_debt']) else "Calculated Net Debt: N/A")
    print(f"Shares Outstanding: {params['calculated_shares_outstanding']:,.0f}" if not pd.isna(params['calculated_shares_outstanding']) else "Shares Outstanding: N/A")
    print("Calculated WACC: N/A (calculation logic removed; AI will be informed)")

    return params

# --- 5. DCF Prompt Generation (Modified: Accepts dcf_params as argument) ---
def generate_fundamental_dcf_prompt(ticker: str, df_with_fundamentals: pd.DataFrame, dcf_params: dict) -> str:
    """
    Generates a detailed prompt for the AI to perform a DCF analysis and fundamental analysis.

    Args:
        ticker (str): The stock ticker symbol.
        df_with_fundamentals (pd.DataFrame): DataFrame containing historical financial data and ratios.
        dcf_params (dict): Dictionary of DCF parameters (e.g., terminal growth rate, WACC, net debt).

    Returns:
        str: The formatted prompt for the AI.
    """
    recent_data = df_with_fundamentals.tail(4)

    # Extract parameters from dcf_params dict
    terminal_growth_rate = dcf_params.get('terminal_growth_rate', np.nan)
    exit_multiple_value = dcf_params.get('exit_multiple_value', np.nan)
    calculated_wacc = dcf_params.get('calculated_wacc', np.nan)
    calculated_net_debt = dcf_params.get('calculated_net_debt', np.nan)
    calculated_shares_outstanding = dcf_params.get('calculated_shares_outstanding', np.nan)

    # Convert numeric parameters to strings for prompt, handling NaN
    terminal_growth_rate_str = f"{terminal_growth_rate * 100:.1f}%" if not pd.isna(terminal_growth_rate) else "NOT PROVIDED"
    exit_multiple_value_str = f"{exit_multiple_value:.1f}x" if not pd.isna(exit_multiple_value) else "NOT PROVIDED"
    wacc_str = f"{calculated_wacc * 100:.1f}%" if not pd.isna(calculated_wacc) else "NOT PROVIDED"
    net_debt_str = f"${calculated_net_debt:,.0f}" if not pd.isna(calculated_net_debt) else "NOT PROVIDED"
    shares_outstanding_str = f"{calculated_shares_outstanding:,.0f}" if not pd.isna(calculated_shares_outstanding) else "NOT PROVIDED"

    # Filter data_cols to only include those present in recent_data to avoid key errors
    data_cols = [col for col in df_with_fundamentals.columns if col not in ['Date'] and not df_with_fundamentals[col].isnull().all()]
    # Limit to a reasonable number of recent periods to avoid overwhelming the LLM
    fundamental_str = recent_data[data_cols].to_string(index=False)

    prompt = rf"""
Perform a Discounted Cash Flow (DCF) analysis for {ticker} using the following recent fundamental financial data:
**Crucially, use ONLY the financial data provided below and the EXPLICITLY GIVEN PARAMETERS. DO NOT assume, estimate, or infer ANY figures or values not explicitly provided. If a required figure for a calculation is not explicitly provided, you MUST state that it cannot be calculated for that specific step and proceed only with what is available.**

Financial Data (DO NOT use any other data):
{fundamental_str}


Given Parameters for DCF (DO NOT use any other values for these metrics):
- **Terminal Growth Rate (g)**: {terminal_growth_rate_str}
- **Exit Multiple**: {exit_multiple_value_str} (EBITDA multiple)
- **Weighted Average Cost of Capital (WACC)**: {wacc_str}
- **Net Debt**: {net_debt_str}
- **Ordinary Shares Number**: {shares_outstanding_str}

Please follow these steps rigorously:

1. **Project Free Cash Flows (FCF)** for the next 5 years.
    * **Strictly use the 'Free_Cash_Flow' data provided in the 'Financial Data' section.**
    * Calculate the average annual growth rate of 'Free_Cash_Flow' from the *provided historical data only*. If there are fewer than two 'Free_Cash_Flow' data points provided or the historical growth cannot be definitively calculated from *only* the provided data, then assume a **constant FCF for projections equal to the latest provided 'Free_Cash_Flow'**.
    * If 'Free_Cash_Flow' is entirely missing or cannot be identified from the provided `fundamental_str`, you MUST state that FCF cannot be projected and cease further DCF calculations.

2. **Calculate the Terminal Value** using both:
    * **Gordon Growth Model (GGM):** Use the formula:
        $ \text{{Terminal Value (GGM)}} = \frac{{\text{{FCF}}_{{\text{{Last Projected Year}}}} * (1 + g)}}{{(\text{{WACC}} - g)}} $
        Use the *provided* Terminal Growth Rate ('g') and WACC. Show the exact numerical calculation with the specific values used from the provided parameters. If FCF for the last projected year, 'g', or 'WACC' is unavailable, state that the Gordon Growth Model cannot be calculated.
    * **Exit Multiple Method (EM):** Use the formula:
        $ \text{{Terminal Value (EM)}} = \text{{EBITDA}}_{{\text{{Last Projected Year}}}} * \text{{Exit Multiple}} $
        Use the *provided* Exit Multiple. Project EBITDA for the last projected year using the average growth rate of 'EBITDA' from the *provided historical data only*. If historical EBITDA growth cannot be definitively calculated, use the latest provided 'EBITDA' as a constant for projection. If 'EBITDA' for the last projected year or the Exit Multiple is unavailable, state that the Exit Multiple Method cannot be calculated.

   Also, calculate the **Weighted Average Cost of Capital (WACC)**. Show formulas and intermediate steps.

3. **Discount all cash flows** (5-year Projected FCFs and Terminal Value) to their Present Value using the *provided* WACC.
    * For each projected FCF_t and for the Terminal Value (TV):
        $ \text{{PV}} = \frac{{\text{{Cash Flow}}_{{t}}}}{{(1 + \text{{WACC}})^t}} $
    * Show the exact mathematical calculation for each discounted cash flow.

4.  **Compute Enterprise Value (EV)** and then determine **Equity Value**:
    * **Enterprise Value (EV)** = Sum of Present Values of 5-year FCFs + Present Value of Terminal Value.
    * **For Terminal Value in EV calculation**: If both GGM and EM are calculable, use the Gordon Growth Model Terminal Value. If only one is calculable, use that one. If neither is calculable, state that EV cannot be calculated.
    * **Equity Value** = Enterprise Value - Net Debt.
    * Use the *provided* Net Debt. If Enterprise Value is not calculable, or if Net Debt is "NOT PROVIDED", state that Equity Value cannot be fully calculated.

5.  **Determine the intrinsic stock price**:
* $ \text{{Intrinsic Price}} = \frac{{\text{{Equity Value}}}}{{\text{{Ordinary Shares Number}}}} $
    * Use the *provided* Ordinary Shares Number. If Equity Value is not calculable, or if Ordinary Shares Number is "NOT PROVIDED", state that the intrinsic stock price cannot be determined.


  **Discounted Cash Flow (DCF) Analysis Summary Table:**
(Populate this table *only* with figures derived directly from your calculations based on the provided data. If any cell cannot be calculated due to missing data or inability to derive from the provided information, fill it with 'N/A' or 'Cannot Calculate'.)

| Year | Projected FCF | Present Value of FCF | Cumulative PV | Terminal Value (Gordon Growth) | Terminal Value (Exit Multiple) | Enterprise Value | Equity Value | Implied Share Price |
|------|----------------|-----------------------|---------------|--------------------------------|--------------------------------|------------------|--------------|---------------------|
|      |                |                       |               |                                |                                |                  |              |                     |
|      |                |                       |               |                                |                                |                  |              |                     |
|      |                |                       |               |                                |                                |                  |              |                     |
|      |                |                       |               |                                |                                |                  |              |                     |
|      |                |                       |               |                                |                                |                  |              |                     |
| TV   | N/A            | [PV of TV]            | N/A           | [Calculated TV GGM]            | [Calculated TV EM]             |                  |              |                     |
| **Summary** | N/A     | N/A                   | N/A           | N/A                            | N/A                            | [Calculated EV]  | [Calculated Equity Value] | [Calculated Intrinsic Price] |

---
**Fundamental Analysis (Ratios):**
Provide a concise fundamental analysis using **only** the following ratios that can be **directly and completely** calculated from the *provided "Financial Data"*. For each ratio discussed, explicitly state the formula used and the numerical input values taken *directly from the provided `fundamental_str`*. Present these insights in clear, descriptive paragraphs. **DO NOT calculate or discuss any ratios for which you lack the direct input figures in the provided data.**
- Net Profit Margin
- Return on Equity (ROE)
- Return on Assets (ROA)
- Current Ratio
- Quick Ratio
- Debt-to-Equity
- Operating Cash Flow Ratio


**Valuation Opinion:**
Offer a clear valuation opinion: Is the stock **undervalued**, **overvalued**, or **fairly valued** relative to its intrinsic value? This opinion **MUST be based strictly on the calculated intrinsic price from the DCF and the insights derived *only* from the provided financial data and calculated ratios.** **DO NOT introduce any external market data (e.g., current market price), external assumptions, or common knowledge about the company or industry.** Explain your reasoning concisely, referring *only* to the results of the DCF and fundamental analysis derived *solely* from the provided data.
"""
    return prompt



# def call_ollama(prompt, model='gemma3'):
#     response = requests.post(
#         'http://localhost:11434/api/generate',
#         json={
#             'model': model,
#             'prompt': prompt,
#             'stream': False  # Set to True if you want streamed output
#         }
#     )
#     return response.json()['response']

# Example usage
#reply = call_ollama(ai_dcf_prompt)
#print(reply)
 
