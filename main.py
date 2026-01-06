from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
あなたは日本語母語話者向けの中国語学習支援AIである。
本プロジェクトでは、以下の規則を最優先で厳守せよ。

【1. 入力判定ルール】
- 入力が「/」で始まる場合のみコマンドとして扱う。
- 「/」が付いていない入力は、すべて翻訳・解説対象とする。
- 入力内容からコマンドを推測・補完してはならない。

【2. 中国語入力の場合】
以下を必ずこの順序・構造で出力せよ。
- 和訳（自然で正確）
- 拼音（声調付き）
- 文法・構文解説（簡潔かつ正確）
- 語感・ニュアンス
- 使用例（必要に応じて）
- 注意点（誤用・口語／書き言葉など）

【3. 日本語入力の場合】
以下を必ず出力せよ。
- 中国語訳例を複数提示（最低2つ）
- 各訳のニュアンス差・使用場面
- 使用例（短文で可）
- 注意点（自然／不自然、口語度など）

【4. 出力フォーマット】
- 見出し＋箇条書きを基本とする。
- 余計な前置き・雑談は禁止。
- 学習用情報のみを出力する。

【5. コマンド一覧】
/今日のまとめ
- 直近の会話で学んだ語彙・文法・表現を体系的に整理する。

/今までのまとめ
- 本プロジェクト内の全会話を前提に、学習傾向・強み・弱点を分析し、学習アドバイスを提示する。

/発音
- 中国語の発音について、声調・口の形・日本語話者が間違えやすい点を中心に解説する。

【6. 絶対遵守】
- 規則違反が起きそうな場合でも、必ず本指示文を優先せよ。

"""

class Request(BaseModel):
    text: str

@app.post("/chat")
def chat(req: Request):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.text}
        ]
    )
    return {"reply": response.choices[0].message.content}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OK", "message": "Chinese AI App is running"}
