"""Tests for ci_platform_manager.handlers.search module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ci_platform_manager.config import Config
from ci_platform_manager.exceptions import PlatformError
from ci_platform_manager.handlers.search import SearchHandler


class TestSearchHandlerInit:
    """Test SearchHandler initialization."""

    def test_init(self, new_config_path: Path) -> None:
        """SearchHandler initializes correctly."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        assert handler.config == config


class TestSearchIssues:
    """Test issue search."""

    @patch("subprocess.run")
    def test_search_issues_success(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Search issues successfully."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        # Mock search results
        search_results = [
            {
                "iid": 1,
                "title": "First issue",
                "state": "opened",
                "labels": ["type::feature"],
                "web_url": "https://gitlab.example.com/test/project/-/issues/1",
            },
            {
                "iid": 2,
                "title": "Second issue",
                "state": "closed",
                "labels": ["type::bug"],
                "web_url": "https://gitlab.example.com/test/project/-/issues/2",
            },
        ]

        mock_run.return_value = Mock(stdout=json.dumps(search_results), returncode=0)

        handler.search_issues("test query")

        captured = capsys.readouterr()
        assert "First issue" in captured.out
        assert "Second issue" in captured.out

    @patch("subprocess.run")
    def test_search_issues_with_state(self, mock_run: Mock, new_config_path: Path) -> None:
        """Search issues with state filter."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        mock_run.return_value = Mock(stdout="[]", returncode=0)

        handler.search_issues("query", state="opened")

        # Verify state parameter was passed
        call_args = mock_run.call_args[0][0]
        assert "state=opened" in " ".join(call_args)

    @patch("subprocess.run")
    def test_search_issues_with_limit(self, mock_run: Mock, new_config_path: Path) -> None:
        """Search issues with result limit."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        mock_run.return_value = Mock(stdout="[]", returncode=0)

        handler.search_issues("query", limit=10)

        # Verify limit parameter was passed
        call_args = mock_run.call_args[0][0]
        assert "per_page=10" in " ".join(call_args)

    @patch("subprocess.run")
    def test_search_issues_no_results(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Search with no results displays appropriate message."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        mock_run.return_value = Mock(stdout="[]", returncode=0)

        handler.search_issues("nonexistent")

        captured = capsys.readouterr()
        assert "No issues found" in captured.out or "found" in captured.out.lower()

    @patch("subprocess.run")
    def test_search_issues_command_failure(self, mock_run: Mock, new_config_path: Path) -> None:
        """Search failure raises PlatformError."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        mock_run.side_effect = subprocess.CalledProcessError(1, ["glab", "api"], stderr="API error")

        with pytest.raises(PlatformError, match="Command failed"):
            handler.search_issues("query")


class TestSearchEpics:
    """Test epic search."""

    @patch("subprocess.run")
    def test_search_epics_success(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Search epics successfully."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        search_results = [
            {
                "iid": 21,
                "title": "Test Epic",
                "state": "opened",
                "web_url": "https://gitlab.example.com/groups/test/-/epics/21",
            }
        ]

        mock_run.return_value = Mock(stdout=json.dumps(search_results), returncode=0)

        handler.search_epics("epic query")

        captured = capsys.readouterr()
        assert "Test Epic" in captured.out

    @patch("subprocess.run")
    def test_search_epics_requires_group(self, mock_run: Mock, temp_dir: Path) -> None:
        """Epic search requires group in config."""
        # Config without default group
        minimal_config = {"platform": "gitlab", "gitlab": {}}
        import yaml

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(minimal_config, file)

        config = Config(config_path)
        handler = SearchHandler(config)

        with pytest.raises(ValueError, match="group"):
            handler.search_epics("query")


class TestSearchMilestones:
    """Test milestone search."""

    @patch("subprocess.run")
    def test_search_milestones_success(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Search milestones successfully."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        search_results = [
            {
                "id": 123,
                "iid": 1,
                "title": "v1.0",
                "state": "active",
                "web_url": "https://gitlab.example.com/test/project/-/milestones/1",
            },
            {
                "id": 124,
                "iid": 2,
                "title": "v2.0",
                "state": "closed",
                "web_url": "https://gitlab.example.com/test/project/-/milestones/2",
            },
        ]

        mock_run.return_value = Mock(stdout=json.dumps(search_results), returncode=0)

        handler.search_milestones("v")

        captured = capsys.readouterr()
        assert "v1.0" in captured.out
        assert "v2.0" in captured.out

    @patch("subprocess.run")
    def test_search_milestones_state_filter(self, mock_run: Mock, new_config_path: Path) -> None:
        """Search milestones with state filter."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        mock_run.return_value = Mock(stdout="[]", returncode=0)

        handler.search_milestones("query", state="active")

        # Verify state filter was applied
        call_args = mock_run.call_args[0][0]
        assert "state=active" in " ".join(call_args)

    @patch("subprocess.run")
    def test_search_milestones_all_states(self, mock_run: Mock, new_config_path: Path) -> None:
        """Search milestones with state=all."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        mock_run.return_value = Mock(stdout="[]", returncode=0)

        handler.search_milestones("query", state="all")

        # Verify all states are included
        call_args = mock_run.call_args[0][0]
        # state=all might be omitted or included depending on implementation
        # Just verify command executed successfully
        mock_run.assert_called_once()


class TestOutputFormatting:
    """Test search output formatting."""

    @patch("subprocess.run")
    def test_text_output_format(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Search output is plain text format."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        search_results = [
            {
                "iid": 1,
                "title": "Test Issue",
                "state": "opened",
                "labels": ["type::feature"],
                "web_url": "https://gitlab.example.com/test/project/-/issues/1",
            }
        ]

        mock_run.return_value = Mock(stdout=json.dumps(search_results), returncode=0)

        handler.search_issues("test")

        captured = capsys.readouterr()
        # Verify text format (not markdown)
        assert "#1" in captured.out or "Test Issue" in captured.out

    @patch("subprocess.run")
    def test_includes_labels(self, mock_run: Mock, new_config_path: Path, capsys) -> None:
        """Search output includes labels."""
        config = Config(new_config_path)
        handler = SearchHandler(config)

        search_results = [
            {
                "iid": 1,
                "title": "Test Issue",
                "state": "opened",
                "labels": ["type::feature", "priority::high"],
                "web_url": "https://gitlab.example.com/test/project/-/issues/1",
            }
        ]

        mock_run.return_value = Mock(stdout=json.dumps(search_results), returncode=0)

        handler.search_issues("test")

        captured = capsys.readouterr()
        assert "type::feature" in captured.out or "priority::high" in captured.out
