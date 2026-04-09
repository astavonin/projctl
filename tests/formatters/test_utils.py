"""Tests for projctl.formatters.utils module."""

import pytest

from projctl.formatters.utils import format_user, format_users


class TestFormatUser:
    """Tests for format_user function."""

    def test_format_user_name_and_username(self) -> None:
        """Full dict renders as 'Name (@username)'."""
        user = {"name": "Alice", "username": "alice"}
        assert format_user(user) == "Alice (@alice)"

    def test_format_user_name_only_no_at_suffix(self) -> None:
        """User with name but no username renders without (@...) suffix."""
        user = {"name": "Alice"}
        assert format_user(user) == "Alice"

    def test_format_user_username_only_falls_back(self) -> None:
        """User with no name falls back to username as display name."""
        user = {"username": "alice"}
        assert format_user(user) == "alice (@alice)"

    def test_format_user_empty_dict(self) -> None:
        """Empty dict returns '?' with no (@) suffix."""
        result = format_user({})
        assert result == "?"


class TestFormatUsers:
    """Tests for format_users function."""

    def test_format_users_empty_list(self) -> None:
        """Empty list returns empty string."""
        assert format_users([]) == ""

    def test_format_users_two_users_comma_separated(self) -> None:
        """Two users are joined with ', '."""
        users = [
            {"name": "Alice", "username": "alice"},
            {"name": "Bob", "username": "bob"},
        ]
        assert format_users(users) == "Alice (@alice), Bob (@bob)"
