"""Tests for projctl.utils.cli_runner module."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from projctl.exceptions import PlatformError
from projctl.utils.cli_runner import run_cli_command


class TestRunCliCommandSuccess:
    """run_cli_command returns stripped stdout on success."""

    @patch("subprocess.run")
    def test_returns_stripped_stdout(self, mock_run: Mock) -> None:
        """Successful command returns stripped stdout."""
        mock_run.return_value = Mock(stdout="  hello world  ", stderr="", returncode=0)

        result = run_cli_command("gh", ["pr", "list"], "gh not found")

        assert result == "hello world"

    @patch("subprocess.run")
    def test_debug_log_emitted(self, mock_run: Mock) -> None:
        """logger.debug is called with the full command."""
        mock_run.return_value = Mock(stdout="output", stderr="", returncode=0)

        with patch("projctl.utils.cli_runner.logger") as mock_logger:
            run_cli_command("gh", ["issue", "list"], "gh not found")
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args[0]
            # The debug message should reference the full command including the binary name
            assert "gh" in str(call_args)


class TestRunCliCommandCalledProcessError:
    """run_cli_command raises PlatformError when the command exits non-zero."""

    @patch("subprocess.run")
    def test_raises_platform_error_with_stderr(self, mock_run: Mock) -> None:
        """CalledProcessError raises PlatformError containing the stderr message."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "pr", "create"], stderr="authentication required"
        )

        with pytest.raises(PlatformError, match="authentication required"):
            run_cli_command("gh", ["pr", "create"], "gh not found")


class TestRunCliCommandFileNotFound:
    """run_cli_command raises PlatformError when the binary is missing."""

    @patch("subprocess.run")
    def test_raises_platform_error_with_not_found_msg(self, mock_run: Mock) -> None:
        """FileNotFoundError raises PlatformError with the provided not_found_msg."""
        not_found_msg = "gh is not installed. Install from https://cli.github.com"
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'gh'")

        with pytest.raises(PlatformError, match="gh is not installed"):
            run_cli_command("gh", ["pr", "list"], not_found_msg)
