from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

# ===== OpenAI client =====
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ===== FastAPI =====
app = FastAPI(title="Chinese AI App")

# ===== System Prompt =====
SYSTEM_PROMPT = """
あなたは中国語学習専用AIである。

【絶対ルール】
1. / で始まらない入力はすべて翻訳・解説対象とする。
2. 中国語が入力された場合：
   - 和訳
   - 拼音
   - 文法・構文解説
   - ニュアンス
   - 使用場面
   - 注意点
   を必ず出力する。
3. 日本語が入力された場合：
   - 複数の中国語訳例
   - それぞれのニュアンス差
   - 使用例
   を必ず出力する。
4. /今日のまとめ 等のコマンドは、/が付いている場合のみ実行する。
5. /の無い文は、どんな場合でも通常解析を行う。

説明は正確・体系的・学習者に最適化すること。
"""

# ===== Request model =====
class TextInput(BaseModel):
    text: str

# ===== Root =====
@app.get("/")
def root():
    return {"status": "OK", "message": "Chinese AI App is running"}

# ===== Main API =====
@app.post("/analyze")
def analyze_text(input: TextInput):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input.text}
        ]
    )

    return {
        "input": input.text,
        "result": response.output_text
    }
