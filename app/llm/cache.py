"""prompt caching 辅助：给 system / tools / 上下文打 cache_control 标记。

Anthropic prompt caching:
- ephemeral 缓存 TTL = 5 分钟
- 单段缓存最小 1024 tokens 才生效
- tools 数组只需在最后一项打标记 = 缓存整段 tools 定义
"""
from __future__ import annotations


def with_cache(text: str) -> list[dict]:
    """把字符串包装成带 cache_control 的 system block 列表。"""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def cache_tools(tools: list[dict]) -> list[dict]:
    """给工具列表最后一项打 cache_control，整段工具定义会被缓存。

    返回新列表（不修改输入）。
    """
    if not tools:
        return tools
    result = [dict(t) for t in tools]
    result[-1] = {**result[-1], "cache_control": {"type": "ephemeral"}}
    return result
