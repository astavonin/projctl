"""Shared formatting utilities for GitLab user display."""

from typing import Any, Dict, List


def format_user(user: Dict[str, Any]) -> str:
    """Format a GitLab user as 'Display Name (@username)'."""
    name = user.get("name") or user.get("username", "?")
    username = user.get("username", "")
    return f"{name} (@{username})" if username else name


def format_users(users: List[Dict[str, Any]]) -> str:
    """Format a list of GitLab users as a comma-separated string."""
    return ", ".join(format_user(u) for u in users)
