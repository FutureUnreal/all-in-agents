[English](README.md) | [中文](README_zh.md)

<p align="center">
  <h1 align="center">all-in-agent</h1>
</p>

<p align="center">
  A minimal, universal agent framework for Python. Zero mandatory dependencies.
</p>

<p align="center">
  <a href="https://pypi.org/project/all-in-agent/"><img src="https://img.shields.io/pypi/v/all-in-agent" alt="PyPI version"></a>
  <a href="https://pypi.org/project/all-in-agent/"><img src="https://img.shields.io/pypi/pyversions/all-in-agent" alt="Python versions"></a>
  <a href="https://pypi.org/project/all-in-agent/"><img src="https://img.shields.io/pypi/l/all-in-agent" alt="License"></a>
  <a href="https://github.com/FutureUnreal/all-in-agent"><img src="https://img.shields.io/github/stars/FutureUnreal/all-in-agent?style=flat" alt="GitHub Stars"></a>
</p>

```bash
pip install all-in-agent
pip install "all-in-agent[openai]"      # OpenAI GPT
pip install "all-in-agent[anthropic]"   # Anthropic Claude
pip install "all-in-agent[all]"         # all optional deps
```

## Why all-in-agent

- 🪶 **Zero dependencies** — pure stdlib core; adapters are opt-in extras
- 🔌 **Pluggable everything** — swap LLM adapter, tools, history, or orchestration without touching other parts
- 🔍 **Transparent by default** — append-only NDJSON event log; every run is replayable
- 🛡️ **Safe by default** — dangerous tools require explicit approval; budget stops runaway agents

## Quick Start

```bash
pip install "all-in-agent[openai]"      # or [anthropic]
```

```python
from all_in_agents import Agent

agent = Agent.quick(model="gpt-4o", workspace=".")
result = agent.run_sync("Summarize README.md in three bullet points")
print(result.final_answer)
```

Or with full control:

```python
from all_in_agents import Agent, OpenAIAdapter, ToolRegistry, BUILTIN_TOOLS, unsafe_defaults

llm = OpenAIAdapter(model="gpt-4o")     # reads OPENAI_API_KEY from env
tools = ToolRegistry(approval_callback=unsafe_defaults())
for t in BUILTIN_TOOLS:                  # read_file, write_file, bash, list_files, text_search
    tools.register(t)

agent = Agent(llm=llm, tools=tools, workspace_root=".")
result = agent.run_sync("Summarize README.md in three bullet points")
print(result.final_answer)
```

> **Jupyter Notebook or async framework?** Use `await agent.run(goal)` directly.

## CLI

```bash
# Single-shot
python -m all_in_agents "Summarize README.md" --model gpt-4o --unsafe

# Interactive REPL
python -m all_in_agents --model gpt-4o --unsafe
```

## Core Concepts

### Node / Flow

Everything is a node. A flow is a graph of nodes.

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

**State contract**: all inter-node state lives in `shared` dict. Node instance fields hold only configuration.

### Budget & Loop Detection

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

### Tool Registry

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

`DANGEROUS` and `WRITES_LOCAL` tools call `approval_callback` before executing. By default, the callback denies all requests (safe by default). Use `unsafe_defaults()` for development or provide your own callback. Install `jsonschema` for automatic argument validation with type coercion.

### History & Compression

`HistoryManager` compresses conversation history when it exceeds 70% of the model's context window (configurable via `compress_threshold_tokens`). It keeps the 12 most recent turns and the 3 most recent tool results verbatim, then asks the LLM to summarize everything older into structured JSON (facts / decisions / open_threads). If LLM summarization fails, it falls back to deterministic snipping.

### Event Store

Every run writes an append-only NDJSON log to `./runs/<run_id>/events.ndjson`:

```
{"event_id": "...", "run_id": "...", "ts": "...", "type": "RUN_CREATED", "payload": {...}}
{"event_id": "...", "run_id": "...", "ts": "...", "type": "ASSISTANT_MESSAGE", "payload": {...}}
{"event_id": "...", "run_id": "...", "ts": "...", "type": "TOOL_RESULT", "payload": {...}}
{"event_id": "...", "run_id": "...", "ts": "...", "type": "RUN_STOPPED", "payload": {"reason": "goal_met"}}
```

### Multi-Agent

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

`TaskManager` uses file-based locking (`fcntl` on Unix, `.lock` file on Windows) for safe concurrent access. Tasks support dependency chains via `dependencies: list[str]`.

## LLM Adapters

| Adapter | Install extra | Environment variable |
|---------|--------------|---------------------|
| `OpenAIAdapter`    | `all-in-agent[openai]`    | `OPENAI_API_KEY`    |
| `AnthropicAdapter` | `all-in-agent[anthropic]` | `ANTHROPIC_API_KEY` |

Both adapters classify errors (TRANSIENT, RATE_LIMITED, AUTH, INVALID_REQUEST, INTERNAL) and retry with exponential backoff. Rate-limited requests honor `retry-after` headers when available.

```python
from all_in_agents import OpenAIAdapter, AnthropicAdapter

llm = OpenAIAdapter(model="gpt-4o-mini", max_retries=3)
llm = AnthropicAdapter(model="claude-sonnet-4-6", max_retries=3)
```

## Architecture

<details>
<summary>📁 Directory Structure</summary>

```
all_in_agents/
├── cli.py       Lightweight CLI runner
├── core/
│   ├── node.py      BaseNode · Node · BatchNode
│   ├── flow.py      Flow (graph runner, auto-retry via exec_with_retry)
│   └── run.py       Run · RunResult · Budget · BudgetExceededError · LoopDetectedError
├── adapters/
│   ├── base.py      LLMAdapter · LLMResponse · ToolCall · LLMError · ErrorClass
│   ├── anthropic.py AnthropicAdapter (error classification, prompt caching)
│   └── openai.py    OpenAIAdapter (error classification, rate-limit tracking)
├── tools/
│   ├── registry.py  ToolRegistry (safe-by-default, approval callbacks, jsonschema)
│   ├── policy.py    ToolPolicy · SideEffectLevel
│   ├── coerce.py    Schema-driven argument type coercion
│   └── builtin.py   read_file · write_file · bash · list_files · text_search
├── history/
│   ├── manager.py   HistoryManager (dynamic threshold, LLM-based compression)
│   ├── compactor.py HistoryCompactor (micro-compact + summarize + fallback)
│   └── store.py     FileEventStore (append-only NDJSON, event callbacks)
└── agents/
    ├── base.py      Agent · Agent.quick() · LLMCallNode · ToolDispatchNode
    ├── harness.py   AGENTS.md / .context/ project context loader
    └── multi.py     MessageBus · TaskManager · MessageEnvelope · Task · TaskStatus
```

</details>

## Package Naming

The PyPI package is `all-in-agent`, but the Python import name is `all_in_agents`:

```bash
pip install all-in-agent
```

```python
from all_in_agents import Agent   # Python import name is 'all_in_agents'
```

The hyphen in the PyPI name can't be used in Python imports, so the module name uses underscores.

## Design Goals

- **Zero mandatory deps** — pure stdlib core; adapters opt-in
- **Small** — ~120 LOC core loop, readable in one sitting
- **Composable** — every piece (Node, Tool, Adapter, History) is replaceable
- **Safe by default** — dangerous tools require approval; budget stops runaway agents

## Requirements

Python 3.10+

Optional: `anthropic`, `openai`, `jsonschema`

## License

MIT
