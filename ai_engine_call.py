import requests
import pandas as pd

def build_prompt(user_preference: str) -> str:
    # companies = company_df.to_string(index=False)
    prompt = f"""
You are a financial AI assistant trained specifically to screen energy sector stocks based strictly on user-defined investment preferences.
Do not ask questions. Do not speculate. Use only the data provided. Always return results in a table.

User request:
"{user_preference}"


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

import requests

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
reply = call_ollama(build_prompt("energy focused and low volatility"))
print(reply)


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
