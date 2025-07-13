import requests
import pandas as pd
import yfinance as yfin

# Get S&P 500 company data
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500 = pd.read_html(url)[0]

# Filter for energy sector companies - doing this for ease of testing 
energy_df = sp500[sp500['GICS Sector'] == 'Energy'][['Symbol', 'Security', 'GICS Sector']]
energy_df.reset_index(drop=True, inplace=True)

energy_df.head()


companies = energy_df.to_string(index=False)  # Convert DataFrame to string for prompt



def build_prompt(user_preference: str, companies) -> str:
    # companies = company_df.to_string(index=False)
    prompt = f"""
You are a financial AI assistant trained specifically to screen energy sector stocks based strictly on user-defined investment preferences.
Do not ask questions. Do not speculate. Use only the data provided. Always return results in a table.
This is not a conversation. Simply return a structured result based on the user input.

User request:
"{user_preference}"

You have access to these energy companies from the S&P 500:
{companies}

Output format requirements:
- 
- Evaluate each company based on relevant factors in the user request.
- Show only the top 3 companies.
- "Rationale" must use bullet points (use `.` for each point)
- Max 3 bullet points per company. 
- Each column must be fixed-width: pad shorter text with spaces so all rows align.
- Do not wrap text. Use one line per row.
- Every cell in a column must be exactly as wide as the longest entry in that column.
- Align using pipes (`|`) and pad with spaces.
- Example formatting (notice aligned columns):


For each company, return:
|----------------------------------------------------------|
| Symbol | Company Name     | Rationale                    |
|--------|------------------|------------------------------|
| ABC    | Alpha Energy     | . Low volatility             |
|        |                  | . Stable returns             |
|        |                  | . High market cap            |
|--------|------------------|------------------------------|

Follow these constraints strictly. No paragraphs, no external data, no assumptions.
"""
    return prompt



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
reply = call_ollama(build_prompt("energy focused and low volatility", companies))
print(reply)

## NEW CODE 
def parse_ai_output(reply):
    # Implement parsing logic here
    # Example: return a list of (symbol, name, bullets)
    return []
# Call the model
reply = call_ollama(build_prompt("energy focused and low volatility", companies))

# Parse AI response
parsed_companies = parse_ai_output(reply)

# Format neatly
formatted_rows = []
for symbol, name, bullets in parsed_companies:
    formatted_rows.extend(format_rationale_row(symbol, name, bullets))

# Show table
print("\n".join(formatted_rows))





#OLD CODE that has been improved above 
# import requests

# def query_llm_stream(prompt: str, model='gemma:latest') -> str:
#     response = requests.post(
#         'http://localhost:11434/api/generate',
#         json={'model': model, 'prompt': prompt, 'stream': True},
#         stream=True
#     )

#     result = ""
#     for line in response.iter_lines():
#         if line:
#             try:
#                 json_line = line.decode("utf-8")
#                 chunk = eval(json_line)  # Better: use json.loads(json_line) if valid JSON
#                 result += chunk.get("response", "")
#             except Exception as e:
#                 print(f"Error parsing line: {e}")
#     return result
