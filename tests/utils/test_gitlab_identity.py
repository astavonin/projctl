"""Tests for projctl.utils.gitlab_identity."""

import pytest

from projctl.exceptions import PlatformError
from projctl.utils.gitlab_identity import CURRENT_USER_QUERY, extract_current_username


class TestExtractCurrentUsername:
    """extract_current_username()'s hard-error contract on a null/absent viewer."""

    def test_returns_username_when_present(self) -> None:
        data = {"currentUser": {"username": "astavonin"}}

        assert extract_current_username(data) == "astavonin"

    def test_null_current_user_raises_platform_error(self) -> None:
        with pytest.raises(PlatformError, match="currentUser returned null"):
            extract_current_username({"currentUser": None})

    def test_missing_current_user_key_raises_platform_error(self) -> None:
        with pytest.raises(PlatformError, match="currentUser returned null"):
            extract_current_username({})

    def test_current_user_without_username_key_raises_platform_error(self) -> None:
        with pytest.raises(PlatformError, match="currentUser returned null"):
            extract_current_username({"currentUser": {}})

    def test_empty_string_username_raises_platform_error(self) -> None:
        """The guard is `if not username`, catching "" the same as None — a
        blank username must never be forwarded as a real identity."""
        with pytest.raises(PlatformError, match="currentUser returned null"):
            extract_current_username({"currentUser": {"username": ""}})


class TestCurrentUserQuery:
    """The callers' argv assertions each interpolate this constant into
    their own expectation, so a rewrite of its selected field would leave
    every one of them green — this pins the literal independently."""

    def test_query_selects_username(self) -> None:
        assert CURRENT_USER_QUERY == "query { currentUser { username } }"
