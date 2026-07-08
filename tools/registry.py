"""Tool registry: maps tool names to callables for plug-and-play routing."""

from typing import Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, description: str):
        self._tools[name] = {"fn": fn, "description": description}

    def get(self, name: str) -> Callable:
        return self._tools[name]["fn"]

    def all_tools(self) -> dict:
        return {k: v["description"] for k, v in self._tools.items()}
