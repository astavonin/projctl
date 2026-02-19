"""Tests for ci_platform_manager.exceptions module."""

import pytest

from ci_platform_manager.exceptions import PlatformError


class TestPlatformError:
    """Test PlatformError exception."""

    def test_platform_error_creation(self) -> None:
        """PlatformError can be created with message."""
        error = PlatformError("Test error message")
        assert str(error) == "Test error message"

    def test_platform_error_raise(self) -> None:
        """PlatformError can be raised and caught."""
        with pytest.raises(PlatformError, match="Test error"):
            raise PlatformError("Test error")

    def test_platform_error_is_exception(self) -> None:
        """PlatformError is an Exception subclass."""
        error = PlatformError("Test")
        assert isinstance(error, Exception)

    def test_platform_error_with_details(self) -> None:
        """PlatformError can include detailed information."""
        details = "Command failed: glab issue create"
        error = PlatformError(f"Failed to create issue: {details}")

        assert "Failed to create issue" in str(error)
        assert "glab issue create" in str(error)
