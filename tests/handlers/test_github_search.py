"""Tests for projctl.handlers.github_search module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from projctl.config import Config
from projctl.exceptions import PlatformError
from projctl.handlers.github_search import GithubSearchHandler


def _make_config(tmp_path: Path) -> Config:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("platform: github\n" "github:\n" "  repo: owner/test-repo\n")
    return Config(cfg_path)


_ISSUES_JSON = json.dumps(
    [
        {
            "number": 1,
            "title": "Found Issue",
            "state": "OPEN",
            "labels": [{"name": "bug"}],
            "url": "https://github.com/owner/test-repo/issues/1",
        }
    ]
)

_MILESTONES_JSON = json.dumps(
    [
        {
            "number": 1,
            "title": "v1.0 Release",
            "state": "open",
            "due_on": None,
            "html_url": "https://github.com/owner/test-repo/milestone/1",
        },
        {
            "number": 2,
            "title": "v2.0 Release",
            "state": "open",
            "due_on": None,
            "html_url": "https://github.com/owner/test-repo/milestone/2",
        },
    ]
)


class TestSearchIssues:
    """Correct gh command constructed for issue search."""

    @patch("subprocess.run")
    def test_search_issues_constructs_correct_command(self, mock_run: Mock, tmp_path: Path) -> None:
        """gh issue list --search called with query and state flags."""
        mock_run.return_value = Mock(stdout=_ISSUES_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        handler = GithubSearchHandler(config)
        results = handler.search_issues("streaming", state="open")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert "issue" in cmd
        assert "list" in cmd
        assert "--search" in cmd
        assert "streaming" in cmd
        assert "--state" in cmd
        assert "open" in cmd
        assert len(results) == 1


class TestSearchMilestones:
    """Client-side milestone filtering by title substring."""

    @patch("subprocess.run")
    def test_search_milestones_filters_by_title(self, mock_run: Mock, tmp_path: Path) -> None:
        """Only milestones whose title contains the query are returned."""
        mock_run.return_value = Mock(stdout=_MILESTONES_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        handler = GithubSearchHandler(config)
        results = handler.search_milestones("v1.0")

        # Only "v1.0 Release" matches
        assert len(results) == 1
        assert results[0]["title"] == "v1.0 Release"

    @patch("subprocess.run")
    def test_search_milestones_case_insensitive_match(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """Query is matched case-insensitively against milestone titles."""
        mock_run.return_value = Mock(stdout=_MILESTONES_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        handler = GithubSearchHandler(config)
        # "V1" uppercase should still match "v1.0 Release"
        results = handler.search_milestones("V1")

        assert len(results) == 1
        assert results[0]["title"] == "v1.0 Release"

    @patch("subprocess.run")
    def test_search_milestones_no_match_returns_empty(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """Query with no match returns an empty list."""
        mock_run.return_value = Mock(stdout=_MILESTONES_JSON, stderr="", returncode=0)

        config = _make_config(tmp_path)
        handler = GithubSearchHandler(config)
        results = handler.search_milestones("v99.0")

        assert results == []

    @patch("subprocess.run")
    def test_search_milestones_returns_empty_on_platform_error(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """PlatformError from gh command yields an empty list (graceful degradation)."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "api"], stderr="forbidden"
        )

        config = _make_config(tmp_path)
        handler = GithubSearchHandler(config)
        results = handler.search_milestones("v1.0")

        assert results == []


class TestSearchIssuesErrorPath:
    """Error path tests for search_issues."""

    @patch("subprocess.run")
    def test_search_issues_returns_empty_on_platform_error(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        """PlatformError from gh command yields an empty list (graceful degradation)."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "issue", "list"], stderr="forbidden"
        )

        config = _make_config(tmp_path)
        handler = GithubSearchHandler(config)
        results = handler.search_issues("streaming")

        assert results == []
