package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
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

	usage = `deepwebseek —— 通过 OpenAI Responses API 调用 DeepSeek（tools 内置 web_search）

用法:
  deepwebseek [选项] [提示词...]
  echo "提示词" | deepwebseek [选项]

提示词:
  优先读取标准输入（管道传入）；否则使用命令行参数拼接。
  选项可以出现在参数中的任意位置。

选项:
  --id <id>          选择配置文件中的模型档案（默认 $DEEPWEBSEEK_ID 或配置文件 default_id）
  --sys <s>          system prompt：内容本身，或指向的文件路径（默认 $DEEPWEBSEEK_SYS）
  --json             请求 JSON 对象输出（response_format 为 {"type":"json_object"}）
  --stream           流式输出
  --model <name>     模型名（默认 $DEEPWEBSEEK_MODEL 或 deepseek-v4-flash）
  --base-url <url>   API 基础地址（默认 $DEEPWEBSEEK_BASE_URL 或 https://api.deepseek.com）
  --max-tokens <n>   最大输出 token 数
  -h, --help         显示本帮助

环境变量:
  DEEPWEBSEEK_API_KEY   可选（也可写在 ~/.deepwebseek.json），DeepSeek API Key
  DEEPWEBSEEK_SYS       可选，system prompt（内容本身或文件路径，可被 --sys 覆盖）
  DEEPWEBSEEK_MODEL     可选，默认模型
  DEEPWEBSEEK_BASE_URL  可选，API 基础地址
  DEEPWEBSEEK_ID        可选，默认模型档案 id

示例:
  deepwebseek "今天北京天气怎么样？" --stream
  deepwebseek "总结这篇文章" --sys "你是资深编辑，输出中文"
  deepwebseek "审查这段代码" --sys ./prompts/code-review.md
  echo "总结一下 OpenAI Responses API 与 web_search" | deepwebseek --stream
  echo '{"问题":"2+2 等于几？"}' | deepwebseek --json

配置文件 ~/.deepwebseek.json（可选，JSON，支持多个模型档案）:
  {"default_id": "deepseek",
   "models": [
     {"id": "deepseek", "model": "deepseek-v4-flash",
      "base_url": "https://api.deepseek.com", "api_key": "sk-..."}
   ]}

优先级（从低到高）:
  默认值 < ~/.deepwebseek.json（按 id 选中的档案） < 环境变量 < 命令行参数
  档案选择：--id > $DEEPWEBSEEK_ID > 配置文件 default_id
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
	sysPrompt string // --sys / $DEEPWEBSEEK_SYS：内容或文件路径
	showHelp  bool
}

func main() {
	opts, err := parseArgs(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "deepwebseek:", err)
		os.Exit(2)
	}
	if opts.showHelp {
		fmt.Print(usage)
		return
	}
	if opts.prompt == "" {
		fmt.Fprintln(os.Stderr, "deepwebseek: 未提供提示词（通过管道传入 stdin，或使用命令行参数）")
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	if opts.apiKey == "" {
		fmt.Fprintln(os.Stderr, "deepwebseek: 缺少 API Key，请设置环境变量 DEEPWEBSEEK_API_KEY 或在 ~/.deepwebseek.json 中配置 api_key")
		os.Exit(2)
	}
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "deepwebseek:", err)
		os.Exit(1)
	}
}

// parseArgs 手工解析参数，允许选项出现在提示词前后任意位置。
// 优先级（从低到高）：默认值 < 配置文件 ~/.deepwebseek.json（按 id 选中的档案）
// < 环境变量 < 命令行参数；档案选择：--id > $DEEPWEBSEEK_ID > 配置文件 default_id。
func parseArgs(args []string) (options, error) {
	o := options{
		model:   defaultModel,
		baseURL: defaultBaseURL,
	}

	// 1. 配置文件 ~/.deepwebseek.json
	cfg, err := loadConfigFile()
	if err != nil {
		return o, err
	}

	// 2. 先解析命令行，记录哪些选项被显式设置（用于最后覆盖）。
	var cli struct {
		id                           string
		model, baseURL, sys          string
		maxTokens                    int
		modelSet, baseURLSet, sysSet bool
		maxTokensSet                 bool
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
		case a == "--id" || strings.HasPrefix(a, "--id="):
			v, err := flagValue(args, &i, a, "--id")
			if err != nil {
				return o, err
			}
			cli.id = v
		case a == "--sys" || strings.HasPrefix(a, "--sys="):
			v, err := flagValue(args, &i, a, "--sys")
			if err != nil {
				return o, err
			}
			cli.sys, cli.sysSet = v, true
		case a == "--model" || strings.HasPrefix(a, "--model="):
			v, err := flagValue(args, &i, a, "--model")
			if err != nil {
				return o, err
			}
			cli.model, cli.modelSet = v, true
		case a == "--base-url" || strings.HasPrefix(a, "--base-url="):
			v, err := flagValue(args, &i, a, "--base-url")
			if err != nil {
				return o, err
			}
			cli.baseURL, cli.baseURLSet = v, true
		case a == "--max-tokens" || strings.HasPrefix(a, "--max-tokens="):
			v, err := flagValue(args, &i, a, "--max-tokens")
			if err != nil {
				return o, err
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return o, fmt.Errorf("--max-tokens 需要整数，得到 %q", v)
			}
			cli.maxTokens, cli.maxTokensSet = n, true
		case strings.HasPrefix(a, "-") && a != "-":
			return o, fmt.Errorf("未知参数 %q（可用 --help 查看用法）", a)
		default:
			promptWords = append(promptWords, a)
		}
	}

	// 3. 选择并应用模型档案：--id > $DEEPWEBSEEK_ID > 配置文件 default_id。
	id := firstNonEmpty(cli.id, os.Getenv("DEEPWEBSEEK_ID"), cfg.DefaultID)
	if err := cfg.applyProfile(&o, id); err != nil {
		return o, err
	}

	// 4. 环境变量覆盖
	if v := os.Getenv("DEEPWEBSEEK_MODEL"); v != "" {
		o.model = v
	}
	if v := os.Getenv("DEEPWEBSEEK_BASE_URL"); v != "" {
		o.baseURL = v
	}
	if v := os.Getenv("DEEPWEBSEEK_API_KEY"); v != "" {
		o.apiKey = v
	}
	if v := os.Getenv("DEEPWEBSEEK_SYS"); v != "" {
		o.sysPrompt = v
	}

	// 5. 命令行参数覆盖（仅覆盖被显式设置的项）
	if cli.modelSet {
		o.model = cli.model
	}
	if cli.baseURLSet {
		o.baseURL = cli.baseURL
	}
	if cli.maxTokensSet {
		o.maxTokens = cli.maxTokens
	}
	if cli.sysSet {
		o.sysPrompt = cli.sys
	}

	// system prompt：若 --sys / $DEEPWEBSEEK_SYS 指向存在的文件则读取其内容，否则视为字面内容。
	sp, err := resolveSysPrompt(o.sysPrompt)
	if err != nil {
		return o, err
	}
	o.sysPrompt = sp

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

// modelProfile 是配置文件中一个模型档案。
type modelProfile struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	BaseURL string `json:"base_url"`
	APIKey  string `json:"api_key"`
}

// configFile 是 ~/.deepwebseek.json 的结构。
type configFile struct {
	DefaultID string         `json:"default_id"`
	Models    []modelProfile `json:"models"`
}

// loadConfigFile 读取 ~/.deepwebseek.json，文件不存在则返回零值。
func loadConfigFile() (configFile, error) {
	var cfg configFile
	home, err := os.UserHomeDir()
	if err != nil {
		return cfg, nil // 无法定位 home 时跳过配置文件
	}
	data, err := os.ReadFile(filepath.Join(home, ".deepwebseek.json"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return cfg, nil
		}
		return cfg, fmt.Errorf("读取配置文件 ~/.deepwebseek.json 失败: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("解析配置文件 ~/.deepwebseek.json 失败: %w", err)
	}
	return cfg, nil
}

// applyProfile 将 id 对应的模型档案应用到 o，零值字段不覆盖；id 为空时不做任何事。
func (c configFile) applyProfile(o *options, id string) error {
	if id == "" {
		return nil
	}
	for _, m := range c.Models {
		if m.ID != id {
			continue
		}
		if m.Model != "" {
			o.model = m.Model
		}
		if m.BaseURL != "" {
			o.baseURL = m.BaseURL
		}
		if m.APIKey != "" {
			o.apiKey = m.APIKey
		}
		return nil
	}
	return fmt.Errorf("配置文件 ~/.deepwebseek.json 中不存在 id 为 %q 的模型档案", id)
}

// firstNonEmpty 返回第一个非空字符串。
func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

// resolveSysPrompt 将 --sys / $DEEPWEBSEEK_SYS 的值解析为 system prompt：
// 值为空返回空；值是可读取的文件路径则返回文件内容；否则视为字面内容原样返回。
func resolveSysPrompt(v string) (string, error) {
	if v == "" {
		return "", nil
	}
	data, err := os.ReadFile(v)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return v, nil
		}
		return "", fmt.Errorf("读取 system prompt 文件 %q 失败: %w", v, err)
	}
	return string(data), nil
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
	if o.sysPrompt != "" {
		params.Instructions = openai.String(o.sysPrompt)
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
		return o.streamOutput(ctx, client, params)
	}
	return o.printOutput(ctx, client, params)
}

// streamOutput 流式输出
func (o options) streamOutput(ctx context.Context, client openai.Client, params responses.ResponseNewParams) error {
	stream := client.Responses.NewStreaming(ctx, params)
	delta := ""
	for stream.Next() {
		ev := stream.Current()
		switch ev.Type {
		case "response.output_text.delta":
			fmt.Print(ev.Delta)
			delta = ev.Delta
		case "response.web_search_call.in_progress":
			if delta != "" && !strings.HasSuffix(delta, "\n") {
				fmt.Println()
			}
			fmt.Fprintf(os.Stderr, gray("web searching [%d]\n"), ev.OutputIndex)
		case "response.web_search_call.searching":
			fmt.Fprintf(os.Stderr, gray("web searching [%d]\n"), ev.OutputIndex)
		case "response.web_search_call.completed":
			fmt.Fprintf(os.Stderr, gray("web searched! [%d]\n"), ev.OutputIndex)
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

// gray 将 s 渲染为灰色前景色文本（ANSI 转义序列）。
func gray(s string) string {
	return "\x1b[90m" + s + "\x1b[0m"
}

// printOutput 非流式输出；--json 模式下先对输出做 Unmarshal/Marshal 规范化。
func (o options) printOutput(ctx context.Context, client openai.Client, params responses.ResponseNewParams) error {
	resp, err := client.Responses.New(ctx, params)
	if err != nil {
		return wrapAPIError(err)
	}
	text := resp.OutputText()
	if strings.TrimSpace(text) == "" {
		// 无输出文本时打印原始响应，方便排查。
		fmt.Fprintln(os.Stderr, "deepwebseek: 响应中没有输出文本，原始响应如下：")
		fmt.Fprintln(os.Stderr, truncate(resp.RawJSON(), 2000))
		return nil
	}
	if o.jsonMode {
		// 先 Unmarshal + Marshal 规范化为紧凑单行 JSON；失败时警告并原样输出。
		var v any
		if err := json.Unmarshal([]byte(text), &v); err != nil {
			fmt.Fprintf(os.Stderr, "deepwebseek: 输出不是合法 JSON，原样打印（%v）\n", err)
			return nil
		}
		if b, err := json.Marshal(v); err != nil {
			fmt.Fprintf(os.Stderr, "deepwebseek: JSON 序列化失败，原样打印（%v）\n", err)
			return nil
		} else {
			text = string(b)
		}
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
