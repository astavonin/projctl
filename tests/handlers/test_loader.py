"""Tests for ci_platform_manager.handlers.loader module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ci_platform_manager.config import Config
from ci_platform_manager.exceptions import PlatformError
from ci_platform_manager.handlers.loader import TicketLoader


class TestTicketLoaderInit:
    """Test TicketLoader initialization."""

    def test_init(self, new_config_path: Path) -> None:
        """Loader initializes correctly."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        assert loader.config == config
        assert loader.group == "test/group"


class TestParseReference:
    """Test reference parsing."""

    def test_parse_issue_number(self, new_config_path: Path) -> None:
        """Parse issue number reference."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("#123")

        assert ref_type == "issue"
        assert ref_id == "123"

    def test_parse_epic_reference(self, new_config_path: Path) -> None:
        """Parse epic reference with & prefix."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("&21")

        assert ref_type == "epic"
        assert ref_id == "21"

    def test_parse_milestone_reference(self, new_config_path: Path) -> None:
        """Parse milestone reference with % prefix."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("%123")

        assert ref_type == "milestone"
        assert ref_id == "123"

    def test_parse_mr_reference(self, new_config_path: Path) -> None:
        """Parse MR reference with ! prefix."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("!134")

        assert ref_type == "mr"
        assert ref_id == "134"

    def test_parse_issue_url(self, new_config_path: Path) -> None:
        """Parse issue URL."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        url = "https://gitlab.example.com/group/project/-/issues/123"
        ref_type, ref_id, project = loader.parse_reference(url)

        assert ref_type == "issue"
        assert ref_id == "123"
        assert project == "group/project"

    def test_parse_epic_url(self, new_config_path: Path) -> None:
        """Parse epic URL."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        url = "https://gitlab.example.com/groups/test/-/epics/21"
        ref_type, ref_id, project = loader.parse_reference(url)

        assert ref_type == "epic"
        assert ref_id == "21"

    def test_parse_mr_url(self, new_config_path: Path) -> None:
        """Parse MR URL."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        url = "https://gitlab.example.com/group/project/-/merge_requests/134"
        ref_type, ref_id, project = loader.parse_reference(url)

        assert ref_type == "mr"
        assert ref_id == "134"

    def test_parse_plain_number(self, new_config_path: Path) -> None:
        """Parse plain number as issue."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("123")

        assert ref_type == "issue"
        assert ref_id == "123"

    def test_parse_invalid_reference(self, new_config_path: Path) -> None:
        """Invalid reference raises ValueError."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        with pytest.raises(ValueError, match="Invalid reference"):
            loader.parse_reference("invalid")


class TestLoadIssue:
    """Test issue loading."""

    @patch("subprocess.run")
    def test_load_issue_success(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str
    ) -> None:
        """Load issue successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, stderr="", returncode=0)

        result = loader.load_issue("#1")

        assert result is not None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_load_issue_with_project(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str
    ) -> None:
        """Load issue with specific project."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        result = loader.load_issue("#1", project="group/project")

        # Verify project was passed in command
        call_args = mock_run.call_args[0][0]
        assert "--repo" in call_args or "group/project" in " ".join(call_args)

    @patch("subprocess.run")
    def test_load_issue_command_failure(self, mock_run: Mock, new_config_path: Path) -> None:
        """Issue loading failure raises PlatformError."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["glab", "issue", "view"], stderr="Error loading issue"
        )

        with pytest.raises(PlatformError, match="Command failed"):
            loader.load_issue("#1")


class TestLoadEpic:
    """Test epic loading."""

    @patch("subprocess.run")
    def test_load_epic_success(
        self, mock_run: Mock, new_config_path: Path, mock_glab_epic_view: str
    ) -> None:
        """Load epic successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        # First call for epic data, second for issues
        mock_run.side_effect = [
            Mock(stdout=mock_glab_epic_view, returncode=0),
            Mock(stdout="[]", returncode=0),  # Empty issues list
        ]

        result = loader.load_epic("&21")

        assert result is not None


class TestLoadMR:
    """Test MR loading."""

    @patch("subprocess.run")
    def test_load_mr_success(
        self, mock_run: Mock, new_config_path: Path, mock_glab_mr_view: str
    ) -> None:
        """Load MR successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_mr_view, returncode=0)

        result = loader.load_mr("!134")

        assert result is not None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_load_mr_with_project(
        self, mock_run: Mock, new_config_path: Path, mock_glab_mr_view: str
    ) -> None:
        """Load MR with specific project."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_mr_view, returncode=0)

        result = loader.load_mr("!134", project="group/project")

        call_args = mock_run.call_args[0][0]
        assert "--repo" in call_args or "group/project" in " ".join(call_args)


class TestLoadMilestone:
    """Test milestone loading."""

    @patch("subprocess.run")
    def test_load_milestone_success(self, mock_run: Mock, new_config_path: Path) -> None:
        """Load milestone successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        milestone_data = {
            "id": 123,
            "iid": 1,
            "title": "v1.0",
            "state": "active",
            "description": "Milestone description",
        }

        # First call for milestone data, second for issues
        mock_run.side_effect = [
            Mock(stdout=json.dumps(milestone_data), returncode=0),
            Mock(stdout="[]", returncode=0),  # Empty issues list
        ]

        result = loader.load_milestone("%123")

        assert result is not None


class TestFormatting:
    """Test output formatting."""

    @patch("subprocess.run")
    def test_markdown_output_format(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, capsys
    ) -> None:
        """Issue output is formatted as markdown."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        loader.load_issue("#1")

        captured = capsys.readouterr()
        # Check markdown formatting
        assert "# " in captured.out or "**" in captured.out

    @patch("subprocess.run")
    def test_includes_metadata(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, capsys
    ) -> None:
        """Output includes issue metadata."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        loader.load_issue("#1")

        captured = capsys.readouterr()
        # Check metadata is included
        assert "State:" in captured.out or "Labels:" in captured.out
