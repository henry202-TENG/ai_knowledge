# MCP (Model Context Protocol)

模型上下文協議，用於標準化 AI 系統與外部工具/數據源的連接。

---

## 1. 什麼是？

### 簡單範例

```
沒有 MCP:
  每個 AI 系統都要寫一套代碼來連接:
  AI → API 1 (自定義代碼)
  AI → API 2 (自定義代碼)
  AI → Database (自定義代碼)

有 MCP:
  AI → MCP Client → MCP Server → 各類資源
  一次開發，到處使用
```

---

## 2. 架構

### 核心組件

```
┌─────────────────────────────────────────┐
│           Application                  │
│  (Claude, GPT, LangChain, etc.)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           MCP Client                    │
│  - Request Builder                      │
│  - Response Parser                      │
└─────────────────┬───────────────────────┘
                  │ JSON-RPC
┌─────────────────▼───────────────────────┐
│           MCP Server                    │
│  - Resource Manager                     │
│  - Tool Executor                        │
│  - Prompt Manager                       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           External Resources           │
│  - APIs, Databases, File Systems        │
└─────────────────────────────────────────┘
```

### 訊息格式

```json
// Client → Server: 調用工具
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "filesystem_read",
    "arguments": {
      "path": "/example/file.txt"
    }
  }
}

// Server → Client: 結果
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": "File content here..."
  }
}
```

---

## 3. 工具定義

### 工具 Schema

```json
{
  "name": "search_codebase",
  "description": "Search for code in the repository",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      },
      "file_pattern": {
        "type": "string",
        "description": "File pattern to search in",
        "default": "*.py"
      },
      "max_results": {
        "type": "number",
        "description": "Maximum results to return",
        "default": 10
      }
    },
    "required": ["query"]
  }
}
```

### 定義工具

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-server")

@app.list_tools()
async def list_tools():
    """列出可用工具"""

    return [
        Tool(
            name="search_codebase",
            description="Search for code in the repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="run_tests",
            description="Run test suite",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Path to tests"
                    }
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name, arguments):
    """執行工具"""

    if name == "search_codebase":
        return search_codebase(arguments["query"])
    elif name == "run_tests":
        return run_tests(arguments["test_path"])
```

---

## 4. 資源管理

### 資源定義

```python
@app.list_resources()
async def list_resources():
    """列出可用資源"""

    return [
        Resource(
            uri="file://config/app.yaml",
            name="app_config",
            description="Application configuration",
            mimeType="application/yaml"
        ),
        Resource(
            uri="db://users",
            name="user_database",
            description="User database",
            mimeType="application/json"
        ),
        Resource(
            uri="api://github/issues",
            name="github_issues",
            description="GitHub issues",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri):
    """讀取資源"""

    if uri.startswith("file://"):
        path = uri.replace("file://", "")
        with open(path) as f:
            return f.read()

    elif uri.startswith("db://"):
        table = uri.replace("db://", "")
        return db.query(f"SELECT * FROM {table}")

    elif uri.startswith("api://"):
        # API 調用
        return fetch_from_api(uri)
```

---

## 5. 提示模板

### 定義提示

```python
@app.list_prompts()
async def list_prompts():
    """列出可用提示模板"""

    return [
        Prompt(
            name="review_code",
            description="Review code changes",
            arguments=[
                PromptArgument(
                    name="pr_url",
                    description="Pull request URL",
                    required=True
                )
            ]
        ),
        Prompt(
            name="analyze_bugs",
            description="Analyze bug reports",
            arguments=[
                PromptArgument(
                    name="time_range",
                    description="Time range to analyze",
                    required=False
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name, arguments):
    """獲取提示"""

    if name == "review_code":
        pr_url = arguments["pr_url"]

        pr_data = fetch_pr(pr_url)

        return ChatMessage(
            role="user",
            content=f"""Review this pull request:

URL: {pr_url}
Title: {pr_data.title}
Author: {pr_data.author}
Files: {pr_data.files}

Please provide a thorough review."""
        )
```

---

## 6. 客戶端使用

### Python 客戶端

```python
from mcp.client import Client

# 連接到 MCP Server
async with Client("http://localhost:3000") as client:

    # 列出工具
    tools = await client.list_tools()
    print(tools)

    # 調用工具
    result = await client.call_tool(
        "search_codebase",
        {"query": "authentication"}
    )

    # 讀取資源
    config = await client.read_resource("file://config/app.yaml")

    # 使用提示
    prompt = await client.get_prompt(
        "review_code",
        {"pr_url": "https://github.com/..."}
    )
```

### LangChain 整合

```python
from langchain_mcp import MCPClient

# 創建 MCP 客戶端
mcp_client = MCPClient("http://localhost:3000")

# 轉換為 LangChain Tools
tools = mcp_client.get_langchain_tools()

# 使用
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
)

result = agent.run("Search for authentication code in the repo")
```

---

## 7. 實作範例

### 完整 Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("example-server")

# Tools
@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="echo",
            description="Echo back the input",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "echo":
        return arguments["message"]

# Resources
@app.list_resources()
async def list_resources():
    return [
        Resource(
            uri="info://server",
            name="server_info",
            description="Server information"
        )
    ]

@app.read_resource()
async def read_resource(uri):
    if uri == "info://server":
        return {"version": "1.0.0", "name": "example-server"}

# Run
if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server.run(app))
```

---

## 8. 最佳實踐

### 設計原則

```
1. 清晰的工具命名
   - 名稱要描述功能
   - 使用動詞開頭

2. 完整的描述
   - 工具描述要清楚用途
   - 參數說明要完整

3. 錯誤處理
   - 定義錯誤格式
   - 提供錯誤恢復建議

4. 安全考慮
   - 驗證輸入
   - 權限控制
   - 審計日誌
```

---

## 9. 與相關技術

| 技術 | 關係 |
|------|------|
| **Function Calling** | 類似的工具調用 |
| **LangChain** | 整合 MCP |
| **OpenAPI** | 類似但更通用 |

---

## 延伸閱讀

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)
- [Anthropic MCP](https://docs.anthropic.com/en/docs/mcp)