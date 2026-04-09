"""Tests for projctl.handlers.github_loader module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from projctl.config import Config
from projctl.exceptions import PlatformError
from projctl.handlers.github_loader import GithubLoader


def _make_config(tmp_path: Path) -> Config:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("platform: github\n" "github:\n" "  repo: owner/test-repo\n")
    return Config(cfg_path)


_ISSUE_JSON = json.dumps(
    {
        "number": 123,
        "title": "Test Issue",
        "body": "# Description\n\nIssue body.",
        "state": "OPEN",
        "labels": [{"name": "bug"}],
        "assignees": [{"login": "octocat"}],
        "milestone": {"title": "v1.0"},
        "url": "https://github.com/owner/test-repo/issues/123",
    }
)

_PR_JSON = json.dumps(
    {
        "number": 123,
        "title": "Test PR",
        "body": "PR body.",
        "state": "OPEN",
        "labels": [],
        "assignees": [],
        "reviewRequests": [],
        "url": "https://github.com/owner/test-repo/pull/123",
        "headRefName": "feature",
        "baseRefName": "main",
    }
)

_MILESTONE_JSON = json.dumps(
    {
        "number": 5,
        "title": "v1.0",
        "state": "open",
        "description": "First release",
        "due_on": None,
        "html_url": "https://github.com/owner/test-repo/milestone/5",
        "open_issues": 2,
        "closed_issues": 3,
    }
)

_MILESTONE_ISSUES_JSON = json.dumps(
    [
        {
            "number": 10,
            "title": "Issue in milestone",
            "state": "OPEN",
            "labels": [],
            "url": "https://github.com/owner/test-repo/issues/10",
        }
    ]
)


class TestLoadIssue:
    """Test load_issue reference parsing and command construction."""

    @patch("subprocess.run")
    def test_load_issue_by_number(self, mock_run: Mock, tmp_path: Path) -> None:
        """gh issue view called with plain number."""
        mock_run.return_value = Mock(stdout=_ISSUE_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        loader.load_issue("123")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert "issue" in cmd
        assert "view" in cmd
        assert "123" in cmd

    @patch("subprocess.run")
    def test_load_issue_by_hash_prefix(self, mock_run: Mock, tmp_path: Path) -> None:
        """#123 prefix is stripped before calling gh."""
        mock_run.return_value = Mock(stdout=_ISSUE_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        loader.load_issue("#123")

        cmd = mock_run.call_args[0][0]
        assert "123" in cmd
        assert "#123" not in " ".join(cmd)


class TestLoadPR:
    """Test load_pr reference parsing."""

    @patch("subprocess.run")
    def test_load_pr_by_bang_prefix(self, mock_run: Mock, tmp_path: Path) -> None:
        """!123 routes to gh pr view 123."""
        mock_run.return_value = Mock(stdout=_PR_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        loader.load_pr("!123")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert "pr" in cmd
        assert "view" in cmd
        assert "123" in cmd


class TestLoadMilestone:
    """Test load_milestone reference parsing."""

    @patch("subprocess.run")
    def test_load_milestone_by_percent_prefix(self, mock_run: Mock, tmp_path: Path) -> None:
        """%5 routes to milestone API then issue list."""

        def side_effect(cmd, **kwargs):
            joined = " ".join(cmd)
            if "milestones/5" in joined:
                return Mock(stdout=_MILESTONE_JSON, stderr="", returncode=0)
            # issue list call
            return Mock(stdout=_MILESTONE_ISSUES_JSON, stderr="", returncode=0)

        mock_run.side_effect = side_effect

        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        result = loader.load_milestone("%5")

        assert result["milestone"]["number"] == 5
        assert len(result["issues"]) == 1

        # Verify milestone API was called with number 5
        milestone_calls = [
            c for c in mock_run.call_args_list if "milestones/5" in " ".join(c[0][0])
        ]
        assert len(milestone_calls) == 1


class TestExtractNumber:
    """Test GithubLoader._extract_number static method."""

    def test_extract_number_hash_prefix(self) -> None:
        """'#123' yields '123'."""
        assert GithubLoader._extract_number("#123", prefix="#") == "123"

    def test_extract_number_plain_string(self) -> None:
        """'123' without prefix yields '123'."""
        assert GithubLoader._extract_number("123", prefix="#") == "123"

    def test_extract_number_bang_prefix(self) -> None:
        """'!134' with '!' prefix yields '134'."""
        assert GithubLoader._extract_number("!134", prefix="!") == "134"

    def test_extract_number_non_digit_raises(self) -> None:
        """Non-digit string raises ValueError."""
        with pytest.raises(ValueError):
            GithubLoader._extract_number("abc", prefix="#")

    def test_extract_number_hash_non_digit_raises(self) -> None:
        """'#abc' raises ValueError after stripping prefix."""
        with pytest.raises(ValueError):
            GithubLoader._extract_number("#abc", prefix="#")

    def test_extract_number_empty_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            GithubLoader._extract_number("", prefix="#")


class TestLoadIssueErrorPath:
    """Error path tests for load_issue."""

    @patch("subprocess.run")
    def test_load_issue_raises_platform_error_on_failure(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """CalledProcessError from subprocess propagates as PlatformError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "issue", "view"], stderr="not found"
        )
        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        with pytest.raises(PlatformError):
            loader.load_issue("123")


class TestLoadPRErrorPath:
    """Error path tests for load_pr."""

    @patch("subprocess.run")
    def test_load_pr_raises_platform_error_on_failure(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """CalledProcessError from subprocess propagates as PlatformError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "pr", "view"], stderr="not found"
        )
        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        with pytest.raises(PlatformError):
            loader.load_pr("123")


class TestLoadMilestoneErrorPath:
    """Error path tests for load_milestone."""

    @patch("subprocess.run")
    def test_load_milestone_raises_platform_error_on_failure(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """CalledProcessError from API call propagates as PlatformError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "api"], stderr="not found"
        )
        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        with pytest.raises(PlatformError):
            loader.load_milestone("5")


class TestLoadMilestoneIssuesGracefulDegradation:
    """load_milestone returns empty issues list when issue fetch fails (TC8)."""

    @patch("subprocess.run")
    def test_load_milestone_issues_empty_on_failure(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """When _load_milestone_issues raises PlatformError, issues list is empty."""

        def side_effect(cmd, **kwargs):
            joined = " ".join(cmd)
            if "milestones/5" in joined:
                return Mock(stdout=_MILESTONE_JSON, stderr="", returncode=0)
            # Simulate issue list failure
            raise subprocess.CalledProcessError(1, cmd, stderr="forbidden")

        mock_run.side_effect = side_effect

        config = _make_config(tmp_path)
        loader = GithubLoader(config)
        result = loader.load_milestone("%5")

        assert result["milestone"]["number"] == 5
        assert result["issues"] == []
