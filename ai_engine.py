import ollama
import pandas as pd

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

def query_llm(prompt: str) -> str:
    response = ollama.chat(
        model='gemma:latest',  # Make sure this matches your local model
        messages=[{'role': 'user', 'content': prompt}],
        stream=False
    )
    return response['message']['content']
from ai_engine import build_prompt, query_llm
import pandas as pd

# User preference
user_pref = "I'm looking for US-based companies with low volatility and good long-term growth."

# Sample company data (replace with your full dataset later)
data = {
    "Symbol": ["OKE", "CVX", "KMI"],
    "Company Name": ["Oneok", "Chevron Corporation", "Kinder Morgan"],
    "Ethics Score": [3, 4, 3],
    "Volatility Score": [4, 3, 3],
    "Growth Score": [4, 4, 4],
}
energy_df = pd.DataFrame(data)

# Build the prompt from user input + data
prompt = build_prompt(user_pref, energy_df)

# Query the LLM (Gemma) via Ollama
output = query_llm(prompt)

# Print result
print(output)


