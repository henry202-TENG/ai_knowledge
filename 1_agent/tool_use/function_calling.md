# Function Calling

## 1. 什麼是？
讓 LLM 能夠調用外部工具或函數的技術，使模型能夠執行 Action 而不僅僅是生成文字。

## 2. 為什麼重要？
- 打破 LLM 只會「說」的限制
- 實現真正的 AI Agent
- 連接真實世界的 API 和資料庫

## 3. 核心原理
- Schema 定義：描述工具的輸入輸出格式
-意圖識別：LLM 判斷何時需要調用工具
- 參數提取：從用戶輸入中提取所需參數
- 結果整合：將工具回傳結果融入對話

## 4. 主流方案
- OpenAI Function Calling
- Anthropic Tool Use
- LangChain Tools
- OpenAI JSON Mode

## 5. 相關主題
- ReAct (推理與行動交替)
- Tool Use 整體框架
- 結構化輸出 (Structured Output)

## 6. 延伸閱讀
- [OpenAI Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

---

*待補充...*