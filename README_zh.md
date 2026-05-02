[English](README.md) | [中文](README_zh.md)

<p align="center">
  <h1 align="center">all-in-agents</h1>
</p>

<p align="center">
  A minimal, universal agent framework for Python. Zero mandatory dependencies.
</p>

<p align="center">
  <a href="https://pypi.org/project/all-in-agents/"><img src="https://img.shields.io/pypi/v/all-in-agents" alt="PyPI version"></a>
  <a href="https://pypi.org/project/all-in-agents/"><img src="https://img.shields.io/pypi/pyversions/all-in-agents" alt="Python versions"></a>
  <a href="https://pypi.org/project/all-in-agents/"><img src="https://img.shields.io/pypi/l/all-in-agents" alt="License"></a>
  <a href="https://github.com/FutureUnreal/all-in-agents"><img src="https://img.shields.io/github/stars/FutureUnreal/all-in-agents?style=flat" alt="GitHub Stars"></a>
</p>

```bash
pip install all-in-agents
pip install "all-in-agents[openai]"      # OpenAI GPT
pip install "all-in-agents[anthropic]"   # Anthropic Claude
pip install "all-in-agents[all]"         # 安装所有可选依赖
```

## 为什么选择 all-in-agents

- 🪶 **零依赖** — 纯标准库核心；适配器按需安装
- 🔌 **全面可插拔** — 随意替换 LLM 适配器、工具、历史记录或编排方式，互不影响
- 🔍 **默认透明** — 仅追加写入的 NDJSON 事件日志，每次运行均可回放
- 🛡️ **默认安全** — 危险工具需要显式审批；预算机制阻止 Agent 失控

## 快速开始

```bash
pip install "all-in-agents[openai]"      # 或 [anthropic]
```

```python
from all_in_agents import Agent, OpenAIAdapter, ToolRegistry, BUILTIN_TOOLS

llm = OpenAIAdapter()                 # 读取环境变量 OPENAI_API_KEY
tools = ToolRegistry()
for t in BUILTIN_TOOLS:               # read_file, write_file, bash
    tools.register(t)

agent = Agent(llm=llm, tools=tools)
result = agent.run_sync("用三个要点总结 README.md")
print(result["final_answer"])
```

> **在 Jupyter Notebook 或异步框架中？** 直接使用 `await agent.run(goal)` 即可。

## 核心概念

### 节点 / 流

一切皆节点。流是节点组成的有向图。

```python
from all_in_agents import BaseNode, Flow

class MyNode(BaseNode):
    async def prep(self, shared: dict):
        return shared["input"]

    async def exec(self, prep_result):
        return prep_result.upper()

    async def post(self, shared: dict, exec_result) -> str:
        shared["output"] = exec_result
        return "default"   # action name → next node

node_a = MyNode()
node_b = MyNode()
node_a >> node_b           # default edge
# or: (node_a - "custom_action") >> node_b

flow = Flow()
await flow.run(shared={}, start=node_a)
```

**状态契约**：所有节点间的状态存储在 `shared` 字典中。节点实例字段仅用于保存配置信息。

### 预算 & 循环检测

```python
from all_in_agents import Budget

budget = Budget(
    max_llm_calls=40,
    max_tool_calls=80,
    max_wall_ms=1_800_000,       # 30 min wall-clock limit
    loop_same_action_limit=3,    # raise LoopDetectedError after 3 consecutive identical tool calls
)

agent = Agent(llm=llm, tools=tools, budget=budget)
```

### Artifact Contracts

当一次运行必须产出可机器检查的文件时，可以使用 artifact contract。
Agent 仍然可以自由执行，但缺少必要产物或产物无效时，框架会把本次运行标记为 `incomplete`。

```python
from all_in_agents import Agent, ArtifactContract

contract = ArtifactContract.files("research_plan.md", "observation.md")

agent = Agent.quick(
    model="gpt-4o",
    workspace=".",
    artifact_contract=contract,
)
result = agent.run_sync("Create the required research artifacts")

assert result.status == "success"
```

安装 `jsonschema` extra 后，JSON 产物也可以做 schema 校验：

```python
contract = ArtifactContract.json_files({
    "metrics.json": {
        "type": "object",
        "required": ["score"],
        "properties": {"score": {"type": "number"}},
    }
})
```

### 工具注册表

```python
from all_in_agents import Tool, ToolRegistry, SideEffectLevel, ToolResponse

async def my_tool(args: dict, run) -> ToolResponse:
    result = do_something(args["input"])
    return ToolResponse(status="success", content=result)

registry = ToolRegistry(
    approval_callback=my_approval_fn   # async (name, args) -> bool
)
registry.register(Tool(
    name="my_tool",
    description="Does something useful",
    input_schema={
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    },
    side_effect_level=SideEffectLevel.READ_ONLY,
    execute=my_tool,
))
```

`DANGEROUS` 级别的工具在执行前会调用 `approval_callback`。安装 `jsonschema` 可启用自动参数校验。

### Skills

项目 skill 是放在 `SKILL.md` 中的提示词包：

```
skills/
  reviewer/
    SKILL.md
.skills/
  local-debug/
    SKILL.md
```

按名称加载指定 skill：

```python
agent = Agent.quick(
    model="gpt-4o",
    workspace=".",
    skills=["reviewer"],
)
```

加载所有发现的 skill：

```python
agent = Agent.quick(model="gpt-4o", workspace=".", skills="all")
```

CLI 用法：

```bash
python -m all_in_agents --skill reviewer "Review this code"
python -m all_in_agents --all-skills "Use the relevant project skill"
python -m all_in_agents --project-context "Follow AGENTS.md and project context"
```

同名 skill 同时存在时，隐藏目录 `.skills/` 优先于 `skills/`。Skills 会注入 system prompt，但不会自动注册 Python 工具。

### 历史记录 & 压缩

`HistoryManager` 会在对话历史超过软阈值时进行压缩。默认阈值是模型上下文窗口的 70%；可以通过 `Agent` 或 `Agent.quick` 的 `history_compress_threshold_tokens` 覆盖。内置 compactor 会以同一个软阈值为目标，保留最近对话，将更早的内容摘要为结构化 JSON（facts / decisions / open_threads），摘要失败时回退到确定性裁剪。

```python
agent = Agent.quick(
    model="gpt-4o",
    history_compress_threshold_tokens=18_000,
)
```

自定义压缩策略可以实现 `compact_turns(llm, turns, *, max_context_tokens, target_tokens=None)` 并返回 `CompactionResult`。

### 事件存储

每次运行会向 `./runs/<run_id>/events.ndjson` 追加写入 NDJSON 日志：

```
{"run_id": "...", "event": "RUN_CREATED", "data": {...}, "ts": "..."}
{"run_id": "...", "event": "ASSISTANT_MESSAGE", "data": {...}, "ts": "..."}
{"run_id": "...", "event": "TOOL_RESULT", "data": {...}, "ts": "..."}
{"run_id": "...", "event": "RUN_STOPPED", "data": {"reason": "goal_met"}, "ts": "..."}
```

### 多智能体

```python
from all_in_agents import MessageBus, TaskManager, MessageEnvelope, Task

bus = MessageBus(run_dir="./runs/session_1")
tm  = TaskManager(run_dir="./runs/session_1")

# coordinator creates tasks
task = await tm.create_task(goal="Analyze file X")

# worker claims and runs
available = await tm.get_available(agent_id="worker_1")
claimed   = await tm.claim_task(available[0].task_id, "worker_1")

# agents communicate
await bus.send(MessageEnvelope(
    msg_id="...", run_id="...",
    from_agent="worker_1", to_agent="coordinator",
    msg_type="TASK_DONE", payload={"result": "..."}, ts="...",
))
```

`TaskManager` 使用基于文件的锁（Unix 上为 `fcntl`，Windows 上为 `.lock` 文件）实现安全的并发访问。任务通过 `dependencies: list[str]` 支持依赖链。

## LLM 适配器

| 适配器 | 安装额外依赖 | 环境变量 |
|--------|-------------|---------|
| `OpenAIAdapter`    | `all-in-agents[openai]`    | `OPENAI_API_KEY`    |
| `AnthropicAdapter` | `all-in-agents[anthropic]` | `ANTHROPIC_API_KEY` |

两个适配器均支持在瞬时错误时以指数退避加抖动方式自动重试。

```python
from all_in_agents import OpenAIAdapter, AnthropicAdapter

llm = OpenAIAdapter(model="gpt-4o-mini", max_retries=3)
llm = AnthropicAdapter(model="claude-sonnet-4-6", max_retries=3)
```

## 架构

<details>
<summary>📁 目录结构</summary>

```
all_in_agents/
├── core/
│   ├── node.py      BaseNode · Node · BatchNode
│   ├── flow.py      Flow (graph runner)
│   └── run.py       Run · Budget · BudgetExceededError · LoopDetectedError
├── adapters/
│   ├── base.py      LLMAdapter · LLMResponse · ToolCall · LLMError · ConfigError
│   ├── anthropic.py AnthropicAdapter (exponential backoff, retry)
│   └── openai.py    OpenAIAdapter
├── tools/
│   ├── registry.py  ToolRegistry (approval callbacks, jsonschema validation)
│   └── builtin.py   read_file · write_file · bash
├── history/
│   ├── manager.py   HistoryManager (LLM-based compression)
│   └── store.py     FileEventStore (append-only NDJSON)
└── agents/
    ├── base.py      Agent · ReActNode · LLMCallNode · ToolDispatchNode
    └── multi.py     MessageBus · TaskManager · MessageEnvelope · Task · TaskStatus
```

</details>

## 包名说明

PyPI 包名为 `all-in-agents`，但 Python import 名为 `all_in_agents`：

```bash
pip install all-in-agents
```

```python
from all_in_agents import Agent   # Python import name is 'all_in_agents'
```

由于 Python 不允许在 import 名中使用连字符（`-`），因此模块名使用下划线 `all_in_agents`。

## 设计目标

- **零强制依赖** — 纯标准库核心；适配器按需引入
- **轻量** — 核心循环约 120 行，一次即可通读
- **可组合** — 每个部分（Node、Tool、Adapter、History）均可替换
- **默认安全** — 危险工具需要审批；预算机制阻止 Agent 失控

## 环境要求

Python 3.10+

可选：`anthropic`、`openai`、`jsonschema`

## 许可证

MIT
