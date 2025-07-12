<<<<<<< HEAD
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ollama

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is running"}


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
        model='gemma3',  # 改为你的模型名称
        messages=[{'role': 'user', 'content': prompt}],
        stream=False
    )
    return response['message']['content']

@app.post("/analyze")
async def analyze_stock(request: Request):
    data = await request.json()
    user_pref = data.get('preference', 'I want stable, US-based energy companies with solid growth.')

    # 示例公司数据（可以改为读取CSV）
    energy_data = {
        "Symbol": ["OKE", "CVX", "KMI", "XOM", "BP"],
        "Company Name": ["Oneok", "Chevron Corporation", "Kinder Morgan", "ExxonMobil", "BP PLC"],
        "Ethics Score": [3, 4, 3, 3, 2],
        "Volatility Score": [4, 3, 3, 2, 4],
        "Growth Score": [4, 4, 4, 5, 3],
    }
    energy_df = pd.DataFrame(energy_data)

    prompt = build_prompt(user_pref, energy_df)
    llm_output = query_llm(prompt)

    return {"reply": llm_output}
=======
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
>>>>>>> 66ad6348bbf35937ff27cdfa212e3ebb8378305b
