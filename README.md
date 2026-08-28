# deepwebseek

通过 OpenAI **官方 Go SDK**（`github.com/openai/openai-go/v3`）的 **Responses 接口**调用 DeepSeek 的 CLI 工具，`tools` 中内置 `web_search`。

## 构建

```bash
go build -o deepwebseek .
```

## 用法

```text
deepseek [选项] [提示词...]
echo "提示词" | deepseek [选项]
```

提示词优先级：

1. **标准输入**（管道/重定向传入），例如 `echo "..." | deepseek`
2. 否则使用**命令行参数**拼接

### 选项

| 选项 | 说明 |
| --- | --- |
| `--json` | 请求 JSON 对象输出（SDK 编码为 `"text":{"format":{"type":"json_object"}}`） |
| `--stream` | 流式输出（`client.Responses.NewStreaming`）；`web_search` 状态打印到 stderr，文本增量输出到 stdout |
| `--model <name>` | 模型名，默认 `$DEEPSEEK_MODEL` 或 `deepseek-v4-flash` |
| `--base-url <url>` | API 基础地址，默认 `$DEEPSEEK_BASE_URL` 或 `https://api.deepseek.com` |
| `--max-tokens <n>` | 最大输出 token 数 |
| `-h, --help` | 显示帮助 |

选项可出现在参数任意位置。

### 环境变量

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 必填，API Key |
| `DEEPSEEK_MODEL` | 可选，默认模型 |
| `DEEPSEEK_BASE_URL` | 可选，API 基础地址 |
