from ai_engine import build_prompt, query_llm
import pandas as pd

# Sample user preference
user_pref = "I'm looking for US-based companies with low volatility and good long-term growth."

# Load company data (replace with your real one)
energy_df = pd.read_csv("data/companies.csv")

# Build prompt and query LLM
prompt = build_prompt(user_pref, energy_df)
response = query_llm(prompt)

print(response)
