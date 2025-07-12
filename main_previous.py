from fastapi import FastAPI
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

@app.get("/analyze")
async def analyze_data():
    # 读取CSV
    df = pd.read_csv('president_heights.csv')
    content = df.to_csv(index=False)

    # 调用Ollama
    response = ollama.chat(
        model='gemma3',  # 如果你的模型名不同，请改
        messages=[
            {'role': 'user', 'content': '请帮我分析以下数据并用Markdown格式回答：\n' + content}
        ]

    )

    reply = response['message']['content']
    return {"reply": reply}
