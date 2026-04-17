#!/usr/bin/env python3
"""PreToolUse hook that blocks direct pip usage and enforces uv workflows."""

import json
import re
import sys
from typing import Any

ALLOWED_UV_PIP_PATTERN = re.compile(r"\buv\s+pip\b", re.IGNORECASE)
PYTHON_MODULE_PIP_PATTERNS = (
    re.compile(r"\bpython(?:\d+(?:\.\d+)*)?\s+-m\s+pip\b", re.IGNORECASE),
    re.compile(r"\bpy(?:\s+-\d+(?:\.\d+)*)?\s+-m\s+pip\b", re.IGNORECASE),
)
DIRECT_PIP_PATTERN = re.compile(r"\bpip3?\b", re.IGNORECASE)

TARGET_TOOLS = {"run_in_terminal", "send_to_terminal", "create_and_run_task"}


def _extract_command_strings(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    commands: list[str] = []

    if tool_name in {"run_in_terminal", "send_to_terminal"}:
        command = tool_input.get("command")
        if isinstance(command, str) and command.strip():
            commands.append(command)

    elif tool_name == "create_and_run_task":
        task = tool_input.get("task")
        if isinstance(task, dict):
            task_command = task.get("command")
            task_args = task.get("args")
            if isinstance(task_command, str) and task_command.strip():
                if isinstance(task_args, list):
                    safe_args = [arg for arg in task_args if isinstance(arg, str)]
                    if safe_args:
                        commands.append(f"{task_command} {' '.join(safe_args)}")
                    else:
                        commands.append(task_command)
                else:
                    commands.append(task_command)

    return commands


def _uses_disallowed_pip(command: str) -> bool:
    normalized = ALLOWED_UV_PIP_PATTERN.sub(" ", command)

    for pattern in PYTHON_MODULE_PIP_PATTERNS:
        if pattern.search(normalized):
            return True

    return bool(DIRECT_PIP_PATTERN.search(normalized))


def _deny_output(reason: str, command: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "systemMessage": f"Blocked command: {command[:200]}",
    }


def _allow_output() -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"systemMessage": f"uv hook warning: invalid input JSON ({exc})"}))
        return 0

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")

    if not isinstance(tool_name, str) or tool_name not in TARGET_TOOLS:
        print(json.dumps(_allow_output()))
        return 0

    if not isinstance(tool_input, dict):
        print(json.dumps(_allow_output()))
        return 0

    commands = _extract_command_strings(tool_name, tool_input)

    for command in commands:
        if _uses_disallowed_pip(command):
            reason = (
                "Direct pip usage is blocked in this repository. "
                "Use uv commands instead, such as 'uv add <package>', "
                "'uv add --dev <package>', 'uv sync', or 'uv run <command>'."
            )
            print(json.dumps(_deny_output(reason, command)))
            return 0

    print(json.dumps(_allow_output()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
