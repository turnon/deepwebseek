# deepwebseek

通过 OpenAI **官方 Go SDK**（`github.com/openai/openai-go/v3`）的 **Responses 接口**调用 DeepSeek 的 CLI 工具，`tools` 中内置 `web_search`。

## 构建

```bash
go build -o deepwebseek .
```

## 用法

```text
deepwebseek [选项] [提示词...]
echo "提示词" | deepwebseek [选项]
```

提示词优先级：

1. **标准输入**（管道/重定向传入），例如 `echo "..." | deepwebseek`
2. 否则使用**命令行参数**拼接

### 选项

| 选项 | 说明 |
| --- | --- |
| `--id <id>` | 选择配置文件 `~/.deepwebseek.json` 中的模型档案（默认 `$DEEPWEBSEEK_ID` 或配置文件 `default_id`） |
| `--sys <s>` | system prompt：值为内容本身，或指向的文件路径（文件存在则读取其内容）；默认 `$DEEPWEBSEEK_SYS` |
| `--json` | 请求 JSON 对象输出（SDK 编码为 `"text":{"format":{"type":"json_object"}}`） |
| `--stream` | 流式输出（`client.Responses.NewStreaming`）；`web_search` 状态打印到 stderr，文本增量输出到 stdout |
| `--model <name>` | 模型名，默认 `$DEEPWEBSEEK_MODEL` 或 `deepseek-flash` |
| `--base-url <url>` | API 基础地址，默认 `$DEEPWEBSEEK_BASE_URL` 或 `https://api.deepseek.com` |
| `--max-tokens <n>` | 最大输出 token 数 |
| `--reasoning <e>` | 推理强度：`none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`（仅对推理模型生效，SDK 编码为 `reasoning.effort`） |
| `--web-search <m>` | web_search 模式：`none`（不启用，不传 `web_search` 工具）/ `auto`（默认，模型自行决定是否搜索）/ `required`（强制搜索，`tool_choice: "required"`） |
| `-h, --help` | 显示帮助 |

选项可出现在参数任意位置。

### 环境变量

| 变量 | 说明 |
| --- | --- |
| `DEEPWEBSEEK_API_KEY` | 可选（也可写在 `~/.deepwebseek.json`），API Key |
| `DEEPWEBSEEK_SYS` | 可选，system prompt（内容本身或文件路径，可被 `--sys` 覆盖） |
| `DEEPWEBSEEK_MODEL` | 可选，默认模型 |
| `DEEPWEBSEEK_BASE_URL` | 可选，API 基础地址 |
| `DEEPWEBSEEK_ID` | 可选，默认模型档案 id |
| `DEEPWEBSEEK_REASONING` | 可选，推理强度（同 `--reasoning` 取值，可被 `--reasoning` 覆盖） |

### system prompt 示例

```bash
# 内容本身
deepwebseek "总结这篇文章" --sys "你是资深编辑，请输出中文"

# 指向文件（读取文件内容作为 system prompt）
deepwebseek "审查这段代码" --sys ./prompts/code-review.md

# 通过环境变量
DEEPWEBSEEK_SYS="你是一个简洁的助手" deepwebseek "什么是 Responses API？"
```

## 配置文件

`~/.deepwebseek.json`（可选，JSON，支持多个模型档案）：

```json
{"default_id": "deepseek",
 "models": [
   {"id": "deepseek", "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com", "api_key": "sk-..."}
 ]}
```

优先级（从低到高）：默认值 < 配置文件（按 id 选中的档案） < 环境变量 < 命令行参数。
档案选择：`--id` > `$DEEPWEBSEEK_ID` > 配置文件 `default_id`。
