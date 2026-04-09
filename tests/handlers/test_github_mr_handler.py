"""Tests for projctl.handlers.github_mr_handler module."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from projctl.exceptions import PlatformError
from projctl.handlers.github_mr_handler import cmd_create_pr


def _args(**kwargs) -> SimpleNamespace:
    """Build a minimal args namespace with sensible defaults."""
    defaults = {
        "title": None,
        "description": None,
        "draft": False,
        "assignee": [],
        "reviewer": [],
        "label": [],
        "milestone": None,
        "target_branch": None,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestCreatePrMinimal:
    """Minimal invocation constructs a valid gh pr create command."""

    @patch("subprocess.run")
    def test_create_pr_minimal(self, mock_run: Mock) -> None:
        """Title-only PR creates correct command."""
        mock_run.return_value = Mock(
            stdout="https://github.com/owner/repo/pull/1", stderr="", returncode=0
        )

        args = _args(title="Add feature X")
        result = cmd_create_pr(args)

        assert result == 0
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert "pr" in cmd
        assert "create" in cmd
        assert "--title" in cmd
        assert "Add feature X" in cmd


class TestCreatePrAllFlags:
    """All supported flags appear in the command when provided."""

    @patch("subprocess.run")
    def test_create_pr_all_flags(self, mock_run: Mock) -> None:
        """title, body, draft, assignee, reviewer, label, milestone, base all present."""
        mock_run.return_value = Mock(
            stdout="https://github.com/owner/repo/pull/2", stderr="", returncode=0
        )

        args = _args(
            title="Full PR",
            description="PR body text",
            draft=True,
            assignee=["octocat"],
            reviewer=["reviewer1"],
            label=["enhancement"],
            milestone="v1.0",
            target_branch="main",
        )
        result = cmd_create_pr(args)

        assert result == 0
        cmd = mock_run.call_args[0][0]
        assert "--title" in cmd
        assert "Full PR" in cmd
        assert "--body" in cmd
        assert "PR body text" in cmd
        assert "--draft" in cmd
        assert "--assignee" in cmd
        assert "octocat" in cmd
        assert "--reviewer" in cmd
        assert "reviewer1" in cmd
        assert "--label" in cmd
        assert "enhancement" in cmd
        assert "--milestone" in cmd
        assert "v1.0" in cmd
        assert "--base" in cmd
        assert "main" in cmd


class TestCreatePrDryRun:
    """Dry-run mode makes no subprocess calls."""

    @patch("subprocess.run")
    def test_create_pr_dry_run(self, mock_run: Mock) -> None:
        """No subprocess.run call in dry-run mode."""
        args = _args(title="Dry PR", dry_run=True)
        result = cmd_create_pr(args)

        assert result == 0
        mock_run.assert_not_called()


class TestCreatePrErrorPaths:
    """Error path tests for cmd_create_pr."""

    @patch("subprocess.run")
    def test_create_pr_returns_1_on_platform_error(self, mock_run: Mock) -> None:
        """PlatformError from run_gh_command returns exit code 1."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "pr", "create"], stderr="authentication required"
        )
        args = _args(title="Error PR")
        result = cmd_create_pr(args)

        assert result == 1

    @patch("subprocess.run")
    def test_create_pr_returns_1_on_os_error(self, mock_run: Mock) -> None:
        """OSError (gh not installed) returns exit code 1."""
        mock_run.side_effect = FileNotFoundError("gh not found")
        args = _args(title="Error PR")
        result = cmd_create_pr(args)

        assert result == 1
