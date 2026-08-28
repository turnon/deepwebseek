// deepseek —— 通过 OpenAI 官方 SDK（openai-go v3）的 Responses 接口调用 DeepSeek 的 CLI 工具。
//
// 使用官方 SDK 的 client.Responses.New / NewStreaming 接口，
// tools 中内置 web_search（{"type": "web_search"}）。
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	defaultBaseURL = "https://api.deepseek.com"
	defaultModel   = "deepseek-v4-flash"

	usage = `deepseek —— 通过 OpenAI Responses API 调用 DeepSeek（tools 内置 web_search）

用法:
  deepseek [选项] [提示词...]
  echo "提示词" | deepseek [选项]

提示词:
  优先读取标准输入（管道传入）；否则使用命令行参数拼接。
  选项可以出现在参数中的任意位置。

选项:
  --json             请求 JSON 对象输出（response_format 为 {"type":"json_object"}）
  --stream           流式输出
  --model <name>     模型名（默认 $DEEPSEEK_MODEL 或 deepseek-v4-flash）
  --base-url <url>   API 基础地址（默认 $DEEPSEEK_BASE_URL 或 https://api.deepseek.com）
  --max-tokens <n>   最大输出 token 数
  -h, --help         显示本帮助

环境变量:
  DEEPSEEK_API_KEY   必填，DeepSeek API Key
  DEEPSEEK_MODEL     可选，默认模型
  DEEPSEEK_BASE_URL  可选，API 基础地址

示例:
  deepseek "今天北京天气怎么样？" --stream
  echo "总结一下 OpenAI Responses API 与 web_search" | deepseek --stream
  echo '{"问题":"2+2 等于几？"}' | deepseek --json
`
)

// options 是命令行解析结果。
type options struct {
	prompt    string // 提示词
	jsonMode  bool   // --json：JSON 对象输出
	stream    bool   // --stream：流式输出
	model     string
	baseURL   string
	apiKey    string
	maxTokens int
	showHelp  bool
}

func main() {
	opts, err := parseArgs(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "deepseek:", err)
		os.Exit(2)
	}
	if opts.showHelp {
		fmt.Print(usage)
		return
	}
	if opts.prompt == "" {
		fmt.Fprintln(os.Stderr, "deepseek: 未提供提示词（通过管道传入 stdin，或使用命令行参数）")
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	if opts.apiKey == "" {
		fmt.Fprintln(os.Stderr, "deepseek: 缺少 API Key，请设置环境变量 DEEPSEEK_API_KEY")
		os.Exit(2)
	}
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "deepseek:", err)
		os.Exit(1)
	}
}

// parseArgs 手工解析参数，允许选项出现在提示词前后任意位置。
func parseArgs(args []string) (options, error) {
	o := options{
		model:   envOr("DEEPSEEK_MODEL", defaultModel),
		baseURL: envOr("DEEPSEEK_BASE_URL", defaultBaseURL),
		apiKey:  os.Getenv("DEEPSEEK_API_KEY"),
	}
	var promptWords []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--json":
			o.jsonMode = true
		case a == "--stream":
			o.stream = true
		case a == "--help" || a == "-h":
			o.showHelp = true
		case a == "--model" || strings.HasPrefix(a, "--model="):
			v, err := flagValue(args, &i, a, "--model")
			if err != nil {
				return o, err
			}
			o.model = v
		case a == "--base-url" || strings.HasPrefix(a, "--base-url="):
			v, err := flagValue(args, &i, a, "--base-url")
			if err != nil {
				return o, err
			}
			o.baseURL = v
		case a == "--max-tokens" || strings.HasPrefix(a, "--max-tokens="):
			v, err := flagValue(args, &i, a, "--max-tokens")
			if err != nil {
				return o, err
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return o, fmt.Errorf("--max-tokens 需要整数，得到 %q", v)
			}
			o.maxTokens = n
		case strings.HasPrefix(a, "-") && a != "-":
			return o, fmt.Errorf("未知参数 %q（可用 --help 查看用法）", a)
		default:
			promptWords = append(promptWords, a)
		}
	}

	// 提示词优先级：有标准输入（管道）则读 stdin，否则用命令行参数。
	if stdinIsPipe() {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return o, fmt.Errorf("读取标准输入失败: %w", err)
		}
		o.prompt = string(data)
	} else {
		o.prompt = strings.Join(promptWords, " ")
	}
	return o, nil
}

// flagValue 支持 "--name=value" 与 "--name value" 两种写法。
func flagValue(args []string, i *int, arg, name string) (string, error) {
	if v, ok := strings.CutPrefix(arg, name+"="); ok {
		return v, nil
	}
	if *i+1 >= len(args) {
		return "", fmt.Errorf("%s 需要一个值", name)
	}
	*i++
	return args[*i], nil
}

// stdinIsPipe 判断标准输入是否为管道/重定向（而非终端）。
func stdinIsPipe() bool {
	info, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice == 0
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// run 用官方 SDK 调用 Responses 接口。
func run(o options) error {
	ctx := context.Background()
	client := openai.NewClient(
		option.WithAPIKey(o.apiKey),
		option.WithBaseURL(strings.TrimRight(o.baseURL, "/")), // SDK 将请求发到 {base}/responses
	)

	// 组装 Responses 请求参数。
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(o.model),
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(o.prompt),
		},
		// 内置 web_search 工具，线上为 {"type": "web_search"}。
		Tools: []responses.ToolUnionParam{
			responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch),
		},
	}
	if o.jsonMode {
		// JSON 对象输出：SDK 将其编码为 "text":{"format":{"type":"json_object"}}。
		params.Text.Format = responses.ResponseFormatTextConfigUnionParam{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	}
	if o.maxTokens > 0 {
		params.MaxOutputTokens = openai.Int(int64(o.maxTokens))
	}

	if o.stream {
		return streamOutput(ctx, client, params)
	}
	return printOutput(ctx, client, params)
}

// ---------- 流式输出 ----------

func streamOutput(ctx context.Context, client openai.Client, params responses.ResponseNewParams) error {
	stream := client.Responses.NewStreaming(ctx, params)
	for stream.Next() {
		ev := stream.Current()
		switch ev.Type {
		case "response.output_text.delta":
			fmt.Print(ev.Delta) // 流式文本，不换行
		case "response.web_search_call.searching":
			fmt.Fprintln(os.Stderr, "🔍 web_search: 搜索中...")
		case "response.web_search_call.in_progress":
			fmt.Fprintln(os.Stderr, "🔍 web_search: 进行中...")
		case "response.web_search_call.completed":
			fmt.Fprintln(os.Stderr, "🔍 web_search: 完成")
		case "error":
			if msg := ev.AsError().Message; msg != "" {
				return errors.New(msg)
			}
		case "response.failed":
			return errors.New("模型响应失败")
		}
	}
	if err := stream.Err(); err != nil {
		return fmt.Errorf("流式请求失败: %w", err)
	}
	fmt.Println() // 流结束后补一个换行
	return nil
}

// ---------- 非流式输出 ----------

func printOutput(ctx context.Context, client openai.Client, params responses.ResponseNewParams) error {
	resp, err := client.Responses.New(ctx, params)
	if err != nil {
		return wrapAPIError(err)
	}
	text := resp.OutputText()
	if strings.TrimSpace(text) == "" {
		// 无输出文本时打印原始响应，方便排查。
		fmt.Fprintln(os.Stderr, "deepseek: 响应中没有输出文本，原始响应如下：")
		fmt.Fprintln(os.Stderr, truncate(resp.RawJSON(), 2000))
		return nil
	}
	fmt.Println(text)
	return nil
}

// wrapAPIError 提取 SDK 返回的 API 错误信息（openai.Error 的 Message）。
func wrapAPIError(err error) error {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		if msg := apiErr.Message; msg != "" {
			return fmt.Errorf("API 错误: %s", msg)
		}
		return fmt.Errorf("API 错误: %v", err)
	}
	return err
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…（已截断）"
}
