"""Tests that CLI cmd_* functions dispatch to the correct platform handler."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from projctl.cli import cmd_create, cmd_load, cmd_search, cmd_create_mr_dispatch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_config(path: Path, platform: str) -> Path:
    cfg_path = path / "config.yaml"
    data: dict = {"platform": platform}
    if platform == "github":
        data["github"] = {"repo": "owner/test-repo"}
        data["common"] = {
            "issue_template": {"required_sections": ["Description", "Acceptance Criteria"]}
        }
    else:
        data["gitlab"] = {
            "default_group": "test/group",
            "labels": {"default": [], "allowed": []},
        }
        data["common"] = {
            "issue_template": {"required_sections": ["Description", "Acceptance Criteria"]}
        }
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return cfg_path


def _write_issues_yaml(path: Path) -> Path:
    issues_path = path / "issues.yaml"
    issues_path.write_text(
        "issues:\n"
        "  - title: Test Issue\n"
        "    description: |\n"
        "      # Description\n\n"
        "      Some details.\n\n"
        "      # Acceptance Criteria\n\n"
        "      - AC1\n"
    )
    return issues_path


def _args(**kwargs) -> SimpleNamespace:
    defaults = {
        "config": None,
        "dry_run": False,
        "yaml_file": None,
        "reference": "1",
        "resource_type": "issue",
        "query": "test",
        "type": "issues",
        "state": "open",
        "limit": 20,
        "title": None,
        "description": None,
        "draft": False,
        "assignee": [],
        "reviewer": [],
        "label": [],
        "milestone": None,
        "target_branch": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# cmd_create
# ---------------------------------------------------------------------------


class TestCmdCreateDispatch:
    """cmd_create selects the correct creator based on platform."""

    def test_cmd_create_github_uses_github_creator(self, tmp_path: Path) -> None:
        """GitHub platform routes to GithubIssueCreator."""
        cfg_path = _write_config(tmp_path, "github")
        issues_path = _write_issues_yaml(tmp_path)
        args = _args(config=str(cfg_path), yaml_file=issues_path, dry_run=True)

        with patch("projctl.cli.GithubIssueCreator") as MockGithubCreator:
            mock_instance = MagicMock()
            MockGithubCreator.return_value = mock_instance
            result = cmd_create(args)

        MockGithubCreator.assert_called_once()
        mock_instance.process_yaml_file.assert_called_once_with(issues_path)
        assert result == 0

    def test_cmd_create_gitlab_uses_epic_creator(self, tmp_path: Path) -> None:
        """GitLab platform routes to EpicIssueCreator."""
        cfg_path = _write_config(tmp_path, "gitlab")
        issues_path = _write_issues_yaml(tmp_path)
        args = _args(config=str(cfg_path), yaml_file=issues_path, dry_run=True)

        with patch("projctl.cli.EpicIssueCreator") as MockEpicCreator:
            mock_instance = MagicMock()
            MockEpicCreator.return_value = mock_instance
            result = cmd_create(args)

        MockEpicCreator.assert_called_once()
        mock_instance.process_yaml_file.assert_called_once_with(issues_path)
        assert result == 0


# ---------------------------------------------------------------------------
# cmd_load
# ---------------------------------------------------------------------------


class TestCmdLoadDispatch:
    """cmd_load selects the correct loader based on platform."""

    def test_cmd_load_github_uses_github_loader(self, tmp_path: Path) -> None:
        """GitHub platform routes to GithubLoader."""
        cfg_path = _write_config(tmp_path, "github")
        args = _args(config=str(cfg_path), resource_type="issue", reference="42")

        with patch("projctl.cli.GithubLoader") as MockLoader:
            mock_instance = MagicMock()
            MockLoader.return_value = mock_instance
            result = cmd_load(args)

        MockLoader.assert_called_once()
        mock_instance.load_issue.assert_called_once_with("42")
        assert result == 0


# ---------------------------------------------------------------------------
# cmd_search
# ---------------------------------------------------------------------------


class TestCmdSearchDispatch:
    """cmd_search selects the correct search handler based on platform."""

    def test_cmd_search_github_uses_github_search_handler(self, tmp_path: Path) -> None:
        """GitHub platform routes to GithubSearchHandler."""
        cfg_path = _write_config(tmp_path, "github")
        args = _args(config=str(cfg_path), type="issues", query="streaming", state="open")

        with patch("projctl.cli.GithubSearchHandler") as MockSearcher:
            mock_instance = MagicMock()
            MockSearcher.return_value = mock_instance
            result = cmd_search(args)

        MockSearcher.assert_called_once()
        mock_instance.search_issues.assert_called_once_with(query="streaming", state="open")
        assert result == 0


# ---------------------------------------------------------------------------
# cmd_create_mr_dispatch
# ---------------------------------------------------------------------------


class TestCmdCreateMrDispatch:
    """cmd_create_mr_dispatch selects the correct MR/PR handler based on platform."""

    def test_cmd_create_mr_github_uses_gh(self, tmp_path: Path) -> None:
        """GitHub platform calls cmd_create_pr with args and config."""
        cfg_path = _write_config(tmp_path, "github")
        args = _args(config=str(cfg_path), title="My PR", dry_run=True)

        with patch("projctl.cli.cmd_create_pr") as mock_pr:
            mock_pr.return_value = 0
            result = cmd_create_mr_dispatch(args)

        mock_pr.assert_called_once()
        call_args = mock_pr.call_args[0]
        assert call_args[0] is args
        assert result == 0
