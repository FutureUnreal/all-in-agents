import asyncio
import os
import re
from pathlib import Path

from .policy import SideEffectLevel
from .registry import Tool, ToolResponse

_MAX_FILE_READ = 100 * 1024  # 100 KB

_MINIMAL_ENV_KEYS = {"PATH", "HOME", "USER", "SHELL", "TERM"}


def _resolve_path(args_path: str, run) -> tuple[Path, str | None]:
    """Resolve a path against workspace_root and check policy constraints."""
    p = Path(args_path)
    if run.workspace_root and not p.is_absolute():
        p = Path(run.workspace_root) / p
    path = p.resolve()

    # Determine allowed roots: from policy or implicit workspace_root
    if run.tool_policy is not None:
        roots = run.tool_policy.resolved_workspace_roots(run.workspace_root)
    elif run.workspace_root:
        roots = (Path(run.workspace_root).resolve(),)
    else:
        roots = ()

    if roots:
        for root in roots:
            if path == root or path.is_relative_to(root):
                return (path, None)
        return (path, "POLICY_BLOCKED")

    return (path, None)


async def _read_file_impl(args: dict, run) -> ToolResponse:
    path, error = _resolve_path(args["path"], run)
    if error:
        return ToolResponse("error", f"Path blocked: {path}", "POLICY_BLOCKED")
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_FILE_READ:
            content = content[:_MAX_FILE_READ] + "\n[TRUNCATED]"
        return ToolResponse("success", content)
    except FileNotFoundError:
        return ToolResponse("error", f"File not found: {args['path']}", "NOT_FOUND")
    except Exception as e:
        return ToolResponse("error", str(e), "INTERNAL_BUG")


async def _write_file_impl(args: dict, run) -> ToolResponse:
    path, error = _resolve_path(args["path"], run)
    if error:
        return ToolResponse("error", f"Path blocked: {path}", "POLICY_BLOCKED")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return ToolResponse("success", f"Written {len(args['content'])} bytes to {args['path']}")
    except Exception as e:
        return ToolResponse("error", str(e), "INTERNAL_BUG")


async def _bash_impl(args: dict, run) -> ToolResponse:
    timeout = int(args.get("timeout", 30))
    command = args["command"]

    if run.tool_policy is not None and run.tool_policy.command_denylist:
        denylist = run.tool_policy.command_denylist
        # Split on shell operators and check each segment's first token
        segments = re.split(r"&&|\|\||;|\|", command)
        for seg in segments:
            token = seg.strip().split()[0] if seg.strip() else ""
            if token and os.path.basename(token) in denylist:
                return ToolResponse("error", f"Command blocked by policy: {token}", "POLICY_BLOCKED")

    env = None
    if run.tool_policy is not None and run.tool_policy.sanitize_env:
        allowed = _MINIMAL_ENV_KEYS | set(run.tool_policy.allowed_env_vars)
        env = {k: v for k, v in os.environ.items() if k in allowed}

    cwd = run.workspace_root if run.workspace_root else None

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResponse("error", f"Command timed out after {timeout}s", "TIMEOUT")

        output = (stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")).strip()
        status = "success" if proc.returncode == 0 else "error"
        return ToolResponse(status, output or f"(exit code {proc.returncode})")
    except Exception as e:
        return ToolResponse("error", str(e), "INTERNAL_BUG")


async def _list_files_impl(args: dict, run) -> ToolResponse:
    base_path_str = args["path"]
    pattern = args.get("pattern", "*")
    max_depth = int(args.get("max_depth", 3))

    base, error = _resolve_path(base_path_str, run)
    if error:
        return ToolResponse("error", f"Path blocked: {base}", "POLICY_BLOCKED")

    if not base.is_dir():
        return ToolResponse("error", f"Not a directory: {base_path_str}", "NOT_FOUND")

    try:
        results = []
        for entry in base.rglob(pattern):
            # Compute depth relative to base
            try:
                rel = entry.relative_to(base)
            except ValueError:
                continue
            depth = len(rel.parts)
            if depth <= max_depth:
                results.append(str(rel))
        results.sort()
        return ToolResponse("success", "\n".join(results))
    except Exception as e:
        return ToolResponse("error", str(e), "INTERNAL_BUG")


async def _text_search_impl(args: dict, run) -> ToolResponse:
    base_path_str = args["path"]
    pattern = args["pattern"]
    max_results = int(args.get("max_results", 50))

    base, error = _resolve_path(base_path_str, run)
    if error:
        return ToolResponse("error", f"Path blocked: {base}", "POLICY_BLOCKED")

    if not base.is_dir():
        return ToolResponse("error", f"Not a directory: {base_path_str}", "NOT_FOUND")

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return ToolResponse("error", f"Invalid pattern: {e}", "INVALID_ARGS")

    try:
        matches = []
        for filepath in sorted(base.rglob("*")):
            if not filepath.is_file():
                continue
            try:
                rel = str(filepath.relative_to(base))
                for line_no, line in enumerate(
                    filepath.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                ):
                    if compiled.search(line):
                        matches.append(f"{rel}:{line_no}:{line}")
                        if len(matches) >= max_results:
                            return ToolResponse("success", "\n".join(matches))
            except Exception:
                continue
        return ToolResponse("success", "\n".join(matches))
    except Exception as e:
        return ToolResponse("error", str(e), "INTERNAL_BUG")


read_file = Tool(
    name="read_file",
    description="Read a file and return its contents.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path to read"}},
        "required": ["path"],
    },
    side_effect_level=SideEffectLevel.READ_ONLY,
    execute=_read_file_impl,
)

write_file = Tool(
    name="write_file",
    description="Write content to a file (creates parent directories as needed).",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    },
    side_effect_level=SideEffectLevel.WRITES_LOCAL,
    execute=_write_file_impl,
)

bash = Tool(
    name="bash",
    description="Execute a shell command and return stdout+stderr.",
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"},
        },
        "required": ["command"],
    },
    side_effect_level=SideEffectLevel.DANGEROUS,
    execute=_bash_impl,
)

list_files = Tool(
    name="list_files",
    description="List files in a directory, optionally filtered by glob pattern and depth.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
            "pattern": {"type": "string", "description": "Glob pattern (default: *)"},
            "max_depth": {"type": "integer", "description": "Max recursion depth (default: 3)"},
        },
        "required": ["path"],
    },
    side_effect_level=SideEffectLevel.READ_ONLY,
    execute=_list_files_impl,
)

text_search = Tool(
    name="text_search",
    description="Search for a regex pattern across files in a directory.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root directory to search"},
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "max_results": {"type": "integer", "description": "Max results to return (default: 50)"},
        },
        "required": ["path", "pattern"],
    },
    side_effect_level=SideEffectLevel.READ_ONLY,
    execute=_text_search_impl,
)

BUILTIN_TOOLS = [read_file, write_file, bash, list_files, text_search]
