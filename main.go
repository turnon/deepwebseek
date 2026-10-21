package main

import (
	"bufio"
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
	"sync"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	defaultBaseURL = "https://api.deepseek.com"
	defaultModel   = "deepseek-flash"

	// 单行提示词的上限；bufio.Scanner 需要显式设置最大 token 大小。
	maxPromptLineSize = 16 << 20 // 16 MiB

	usage = `deepwebseek —— 通过 OpenAI Responses API 调用 DeepSeek（tools 内置 web_search）

用法:
  deepwebseek [选项] [提示词...]
  echo "提示词" | deepwebseek [选项]

提示词:
  使用 --prompts 时，文件中的每一行都是一个独立的提示词；
  否则优先读取标准输入（管道传入），再否则使用命令行参数拼接。
  选项可以出现在参数中的任意位置。

选项:
  --id <id>          选择配置文件中的模型档案（默认 $DEEPWEBSEEK_ID 或配置文件 default_id）
  --sys <s>          system prompt：内容本身，或指向的文件路径（默认 $DEEPWEBSEEK_SYS）
  --json             请求 JSON 对象输出（response_format 为 {"type":"json_object"}）
  --stream           流式输出（仅单条提示词；--prompts 批量模式不使用）
  --prompts <file>   从文件逐行读取提示词，每行一个
  --prompts-parallel <n>
                     并发发送 --prompts 中请求的数量（默认 1）
  --model <name>     模型名（默认 $DEEPWEBSEEK_MODEL 或 deepseek-flash）
  --base-url <url>   API 基础地址（默认 $DEEPWEBSEEK_BASE_URL 或 https://api.deepseek.com）
  --max-tokens <n>   最大输出 token 数
  --reasoning <e>    推理强度：none/minimal/low/medium/high/xhigh/max（仅对推理模型生效）
  --web-search <m>   web_search 模式：none（不启用）/ auto（默认，模型自行决定）/ required（必须搜索）

环境变量:
  DEEPWEBSEEK_API_KEY   可选（也可写在 ~/.deepwebseek.json），DeepSeek API Key
  DEEPWEBSEEK_SYS       可选，system prompt（内容本身或文件路径，可被 --sys 覆盖）
  DEEPWEBSEEK_MODEL     可选，默认模型
  DEEPWEBSEEK_BASE_URL  可选，API 基础地址
  DEEPWEBSEEK_ID        可选，默认模型档案 id
  DEEPWEBSEEK_REASONING 可选，推理强度（同 --reasoning 取值，可被 --reasoning 覆盖）

示例:
  deepwebseek "今天北京天气怎么样？" --stream
  deepwebseek "总结这篇文章" --sys "你是资深编辑，输出中文"
  deepwebseek "审查这段代码" --sys ./prompts/code-review.md
  deepwebseek --prompts ./prompts.txt --prompts-parallel 4
  echo "总结一下 OpenAI Responses API 与 web_search" | deepwebseek --stream
  echo '{"问题":"2+2 等于几？"}' | deepwebseek --json

配置文件 ~/.deepwebseek.json（可选，JSON，支持多个模型档案）:
  {"default_id": "deepseek",
   "models": [
     {"id": "deepseek", "model": "deepseek-flash",
      "base_url": "https://api.deepseek.com", "api_key": "sk-..."}
   ]}

优先级（从低到高）:
  默认值 < ~/.deepwebseek.json（按 id 选中的档案） < 环境变量 < 命令行参数
  档案选择：--id > $DEEPWEBSEEK_ID > 配置文件 default_id
`
)

// options 是命令行解析结果。
type options struct {
	prompt          string // 提示词
	promptsFile     string // --prompts：提示词文件路径（逐行读取）
	promptsParallel int    // --prompts-parallel：并发请求数
	jsonMode        bool   // --json：JSON 对象输出
	stream          bool   // --stream：流式输出
	model           string
	baseURL         string
	apiKey          string
	maxTokens       int
	sysPrompt       string                 // --sys / $DEEPWEBSEEK_SYS：内容或文件路径
	reasoning       shared.ReasoningEffort // 推理强度：none/minimal/low/medium/high/xhigh/max
	webSearch       string                 // --web-search：none/auto/required，空值相当于 auto
	showHelp        bool
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
	if opts.promptsFile == "" && opts.prompt == "" {
		fmt.Fprintln(os.Stderr, "deepwebseek: 未提供提示词（通过管道传入 stdin、使用命令行参数，或使用 --prompts 指定文件）")
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
		model:           defaultModel,
		baseURL:         defaultBaseURL,
		promptsParallel: 1,
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
		prompts                      string
		maxTokens                    int
		reasoning                    shared.ReasoningEffort
		webSearch                    string
		modelSet, baseURLSet, sysSet bool
		promptsSet                   bool
		maxTokensSet, reasoningSet   bool
		webSearchSet                 bool
	}
	var promptWords []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--json":
			o.jsonMode = true
		case a == "--stream":
			o.stream = true
		case a == "--prompts" || strings.HasPrefix(a, "--prompts="):
			v, err := flagValue(args, &i, a, "--prompts")
			if err != nil {
				return o, err
			}
			cli.prompts, cli.promptsSet = v, true
		case a == "--prompts-parallel" || strings.HasPrefix(a, "--prompts-parallel="):
			v, err := flagValue(args, &i, a, "--prompts-parallel")
			if err != nil {
				return o, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return o, fmt.Errorf("--prompts-parallel 需要正整数，得到 %q", v)
			}
			o.promptsParallel = n
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
		case a == "--reasoning" || strings.HasPrefix(a, "--reasoning="):
			v, err := flagValue(args, &i, a, "--reasoning")
			if err != nil {
				return o, err
			}
			cli.reasoning, err = parseReasoning(v)
			if err != nil {
				return o, err
			}
			cli.reasoningSet = true
		case a == "--web-search" || strings.HasPrefix(a, "--web-search="):
			v, err := flagValue(args, &i, a, "--web-search")
			if err != nil {
				return o, err
			}
			cli.webSearch, err = parseWebSearchMode(v)
			if err != nil {
				return o, err
			}
			cli.webSearchSet = true
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
	if v := os.Getenv("DEEPWEBSEEK_REASONING"); v != "" {
		e, err := parseReasoning(v)
		if err != nil {
			return o, err
		}
		o.reasoning = e
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
	if cli.reasoningSet {
		o.reasoning = cli.reasoning
	}
	if cli.webSearchSet {
		o.webSearch = cli.webSearch
	}

	// system prompt：若 --sys / $DEEPWEBSEEK_SYS 指向存在的文件则读取其内容，否则视为字面内容。
	sp, err := resolveSysPrompt(o.sysPrompt)
	if err != nil {
		return o, err
	}
	o.sysPrompt = sp

	// 提示词优先级：--prompts 文件 > 标准输入（管道） > 命令行参数。
	if cli.promptsSet {
		o.promptsFile = cli.prompts
	} else if stdinIsPipe() {
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

// parseWebSearchMode 解析 --web-search 的取值。
// 返回 none/auto/required；大小写不敏感，非法取值返回错误。
func parseWebSearchMode(s string) (string, error) {
	switch strings.ToLower(s) {
	case "none":
		return "none", nil
	case "auto":
		return "auto", nil
	case "required":
		return "required", nil
	default:
		return "", fmt.Errorf("--web-search 取值无效 %q（可选: none auto required）", s)
	}
}

// parseReasoning 解析 --reasoning / $DEEPWEBSEEK_REASONING 的推理强度取值。
// 大小写不敏感，返回 SDK 中的 shared.ReasoningEffort；非法取值返回错误。
func parseReasoning(s string) (shared.ReasoningEffort, error) {
	switch strings.ToLower(s) {
	case "none":
		return shared.ReasoningEffortNone, nil
	case "minimal":
		return shared.ReasoningEffortMinimal, nil
	case "low":
		return shared.ReasoningEffortLow, nil
	case "medium":
		return shared.ReasoningEffortMedium, nil
	case "high":
		return shared.ReasoningEffortHigh, nil
	case "xhigh":
		return shared.ReasoningEffortXhigh, nil
	case "max":
		return shared.ReasoningEffortMax, nil
	default:
		return "", fmt.Errorf("--reasoning 取值无效 %q（可选: none minimal low medium high xhigh max）", s)
	}
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

	// 每个输出方法都基于 options 构造自己的请求参数，避免复用同一个 params。
	if o.promptsFile != "" {
		return o.parallelOutput(ctx, client)
	}
	if o.stream {
		return o.streamOutput(ctx, client)
	}
	return o.printOutput(ctx, client)
}

// newResponseParams 基于 o 当前字段和 prompt 构造一份全新的 Responses 请求参数。
// 每次请求都应调用它，避免在并发路径中复用同一个 params。
func (o options) newResponseParams(prompt string) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(o.model),
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(prompt),
		},
	}
	// 内置 web_search 工具，线上为 {"type": "web_search"}；--web-search none 时不附加。
	if o.webSearch != "none" {
		params.Tools = append(params.Tools,
			responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch))
	}
	// --web-search required：强制模型调用 web_search（tool_choice: "required"）。
	if o.webSearch == "required" {
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		}
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
	if o.reasoning != "" {
		params.Reasoning.Effort = o.reasoning
	}
	return params
}

// streamOutput 流式输出。
func (o options) streamOutput(ctx context.Context, client openai.Client) error {
	params := o.newResponseParams(o.prompt)
	stream := client.Responses.NewStreaming(ctx, params)
	delta := ""
	inReasoning := false // 是否正在输出思考过程
	for stream.Next() {
		ev := stream.Current()
		switch ev.Type {
		case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
			// 思考过程走 stderr 并置灰，不污染 stdout 的正文输出
			fmt.Fprint(os.Stderr, gray(ev.Delta))
			inReasoning = true
		case "response.output_text.delta":
			if inReasoning {
				// 思考结束、正文开始，补换行 + 分隔线
				fmt.Fprintln(os.Stderr)
				fmt.Fprintln(os.Stderr, gray("---"))
				inReasoning = false
			}
			fmt.Print(ev.Delta)
			delta = ev.Delta
		case "response.web_search_call.in_progress":
			if inReasoning {
				fmt.Fprintln(os.Stderr)
				inReasoning = false
			}
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

// justGetResponse 请求 Responses 接口并返回输出文本。
// --json 模式下会先把输出规范化为紧凑 JSON；没有输出文本时打印原始响应便于排查。
func (o options) justGetResponse(ctx context.Context, client openai.Client, prompt string) (string, error) {
	params := o.newResponseParams(prompt)
	resp, err := client.Responses.New(ctx, params)
	if err != nil {
		return "", wrapAPIError(err)
	}
	text := resp.OutputText()
	if strings.TrimSpace(text) == "" {
		// 无输出文本时打印原始响应，方便排查。
		fmt.Fprintln(os.Stderr, "deepwebseek: 响应中没有输出文本，原始响应如下：")
		fmt.Fprintln(os.Stderr, truncate(resp.RawJSON(), 2000))
		return "", nil
	}
	if o.jsonMode {
		// 先 Unmarshal + Marshal 规范化为紧凑单行 JSON；失败时警告并原样输出。
		var v any
		if err := json.Unmarshal([]byte(text), &v); err != nil {
			fmt.Fprintf(os.Stderr, "deepwebseek: 输出不是合法 JSON，原样打印（%v）\n", err)
			return text, nil
		}
		b, err := json.Marshal(v)
		if err != nil {
			fmt.Fprintf(os.Stderr, "deepwebseek: JSON 序列化失败，原样打印（%v）\n", err)
			return text, nil
		}
		text = string(b)
	}
	return text, nil
}

// printOutput 非流式输出。
func (o options) printOutput(ctx context.Context, client openai.Client) error {
	text, err := o.justGetResponse(ctx, client, o.prompt)
	if err != nil {
		return err
	}
	if text != "" {
		fmt.Println(text)
	}
	return nil
}

// parallelOutput 逐行读取 promptsFile 并并发发送每个提示词，并发数由 o.promptsParallel 控制。
// 文件内容是流式读取的，不会一次性全部载入内存。
// 每个请求完成后整体打印响应，因此 --stream 不适用于批量模式。
func (o options) parallelOutput(ctx context.Context, client openai.Client) error {
	promptsFile := o.promptsFile
	f, err := os.Open(promptsFile)
	if err != nil {
		return fmt.Errorf("打开提示词文件 %q 失败: %w", promptsFile, err)
	}
	defer f.Close()

	workers := o.promptsParallel
	if workers < 1 {
		workers = 1
	}

	type promptJob struct {
		index  int
		prompt string
	}
	jobs := make(chan promptJob)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var outputMu sync.Mutex
	var firstErr error
	recordErr := func(err error) {
		mu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		mu.Unlock()
	}

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				// 每个提示词单独构造请求参数，避免并发复用同一个 params。
				text, err := o.justGetResponse(ctx, client, job.prompt)
				if err != nil {
					recordErr(fmt.Errorf("提示词 #%d: %w", job.index+1, err))
					continue
				}
				if text != "" {
					// 保证每个响应整体输出，避免并发打印互相穿插。
					outputMu.Lock()
					fmt.Println(text)
					outputMu.Unlock()
				}
			}
		}()
	}

	// 边读边发送：jobs 无缓冲，读取速度会受并发请求数限制，内存占用保持稳定。
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), maxPromptLineSize)
	count := 0
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		select {
		case jobs <- promptJob{index: count, prompt: line}:
			count++
		case <-ctx.Done():
			recordErr(ctx.Err())
			close(jobs)
			wg.Wait()
			return firstErr
		}
	}
	if err := scanner.Err(); err != nil {
		recordErr(fmt.Errorf("读取提示词文件 %q 失败: %w", promptsFile, err))
	}
	close(jobs)
	wg.Wait()

	if count == 0 && firstErr == nil {
		return fmt.Errorf("提示词文件 %q 中没有有效提示词", promptsFile)
	}
	return firstErr
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
