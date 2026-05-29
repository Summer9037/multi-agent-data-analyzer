"""Anthropic Claude 客户端单例。"""
from __future__ import annotations

from anthropic import Anthropic

from app.config import settings

_client: Anthropic | None = None


def get_client() -> Anthropic:
    """返回进程内单例 Anthropic 客户端。"""
    global _client
    if _client is None:
        kwargs: dict = {"api_key": settings.anthropic_api_key}
        if settings.anthropic_base_url:
            kwargs["base_url"] = settings.anthropic_base_url
        _client = Anthropic(**kwargs)
    return _client
