"""Tests for ci_platform_manager.handlers.creator module."""

import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import yaml

from ci_platform_manager.config import Config
from ci_platform_manager.exceptions import PlatformError
from ci_platform_manager.handlers.creator import EpicIssueCreator


class TestEpicIssueCreatorInit:
    """Test EpicIssueCreator initialization."""

    def test_init_normal_mode(self, new_config_path: Path) -> None:
        """Creator initializes correctly in normal mode."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config, dry_run=False)

        assert creator.config == config
        assert creator.dry_run is False
        assert creator.group == "test/group"
        assert creator.created_issues == []
        assert creator.issue_id_mapping == {}

    def test_init_dry_run_mode(self, new_config_path: Path) -> None:
        """Creator initializes correctly in dry-run mode."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config, dry_run=True)

        assert creator.dry_run is True


class TestCreateEpic:
    """Test epic creation."""

    @patch("subprocess.run")
    def test_create_new_epic(self, mock_run: Mock, new_config_path: Path) -> None:
        """Create new epic via API call."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock successful epic creation
        mock_run.return_value = Mock(stdout='{"id": 456, "iid": 21}', stderr="", returncode=0)

        epic_config = {"title": "Test Epic", "description": "Epic description"}

        result = creator.create_epic(epic_config)

        # Verify glab api was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "glab"
        assert "api" in call_args

    def test_use_existing_epic(self, new_config_path: Path) -> None:
        """Use existing epic ID without API call."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        epic_config = {"id": 21}

        result = creator.create_epic(epic_config)

        assert result == "21"

    def test_epic_missing_required_fields(self, new_config_path: Path) -> None:
        """Epic without id or title raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        epic_config = {"description": "Missing title"}

        with pytest.raises(ValueError, match="must have either 'id' or 'title'"):
            creator.create_epic(epic_config)

    @patch("subprocess.run")
    def test_create_epic_dry_run(self, mock_run: Mock, new_config_path: Path) -> None:
        """Dry run mode doesn't execute commands."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config, dry_run=True)

        epic_config = {"title": "Test Epic", "description": "Epic description"}

        result = creator.create_epic(epic_config)

        # No actual command execution in dry run
        mock_run.assert_not_called()


class TestCreateIssue:
    """Test issue creation."""

    @patch("subprocess.run")
    def test_create_issue_minimal(self, mock_run: Mock, new_config_path: Path) -> None:
        """Create issue with minimal required fields."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock successful issue creation
        mock_run.return_value = Mock(
            stdout="https://gitlab.example.com/test/project/-/issues/1", stderr="", returncode=0
        )

        issue_config = {
            "title": "Test Issue",
            "description": "# Description\n\nTest description\n\n# Acceptance Criteria\n\n- AC1",
        }

        creator.create_issue(issue_config, epic_id=None)

        # Verify glab issue create was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "glab"
        assert "issue" in call_args
        assert "create" in call_args

    @patch("subprocess.run")
    def test_create_issue_with_metadata(self, mock_run: Mock, new_config_path: Path) -> None:
        """Create issue with all metadata fields."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        mock_run.return_value = Mock(
            stdout="https://gitlab.example.com/test/project/-/issues/1", returncode=0
        )

        issue_config = {
            "title": "Test Issue",
            "description": "# Description\n\nTest\n\n# Acceptance Criteria\n\n- AC1",
            "labels": ["type::bug"],
            "assignee": "testuser",
            "milestone": "v1.0",
            "due_date": "2026-03-01",
        }

        creator.create_issue(issue_config, epic_id=None)

        # Verify metadata was passed in command
        call_args = mock_run.call_args[0][0]
        assert "--assignee" in call_args
        assert "testuser" in call_args
        assert "--milestone" in call_args
        assert "v1.0" in call_args

    @patch("subprocess.run")
    def test_create_issue_command_failure(self, mock_run: Mock, new_config_path: Path) -> None:
        """Issue creation failure raises PlatformError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["glab", "issue", "create"], stderr="Error creating issue"
        )

        issue_config = {
            "title": "Test Issue",
            "description": "# Description\n\nTest\n\n# Acceptance Criteria\n\n- AC1",
        }

        with pytest.raises(PlatformError, match="Command failed"):
            creator.create_issue(issue_config, epic_id=None)


class TestLoadYAML:
    """Test YAML file loading."""

    def test_load_valid_yaml(self, new_config_path: Path, sample_issue_yaml_path: Path) -> None:
        """Load valid YAML file successfully in dry-run mode."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config, dry_run=True)

        # Test in dry-run mode to avoid complex mocking
        creator.process_yaml_file(sample_issue_yaml_path)

        # In dry-run mode, should track created issues
        assert len(creator.created_issues) == 2

    def test_load_nonexistent_yaml(self, new_config_path: Path, temp_dir: Path) -> None:
        """Loading nonexistent file raises FileNotFoundError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        nonexistent = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            creator.process_yaml_file(nonexistent)

    def test_load_invalid_yaml(self, new_config_path: Path, temp_dir: Path) -> None:
        """Loading invalid YAML raises appropriate error."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [[[")

        with pytest.raises(yaml.YAMLError):
            creator.process_yaml_file(invalid_yaml)


class TestValidation:
    """Test validation methods."""

    def test_validate_labels_success(self, new_config_path: Path) -> None:
        """Valid labels pass validation."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        labels = ["type::feature", "type::bug"]

        # Should not raise
        creator._validate_issue_labels(labels)

    def test_validate_labels_unknown(self, new_config_path: Path) -> None:
        """Unknown labels raise ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        labels = ["unknown::label"]

        with pytest.raises(ValueError, match="Unknown labels"):
            creator._validate_issue_labels(labels)

    def test_validate_description_success(self, new_config_path: Path) -> None:
        """Valid description passes validation."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        issue_config = {
            "title": "Test Issue",
            "description": "# Description\n\nContent\n\n# Acceptance Criteria\n\n- AC1",
        }

        # Should not raise
        creator._validate_issue_description(issue_config)

    def test_validate_description_missing_sections(self, new_config_path: Path) -> None:
        """Description missing required sections raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        issue_config = {
            "title": "Test Issue",
            "description": "# Description\n\nOnly description",
        }

        with pytest.raises(ValueError, match="missing required sections"):
            creator._validate_issue_description(issue_config)


class TestDryRun:
    """Test dry-run mode."""

    @patch("subprocess.run")
    def test_dry_run_no_execution(self, mock_run: Mock, new_config_path: Path) -> None:
        """Dry run mode prevents command execution."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config, dry_run=True)

        issue_config = {
            "title": "Test Issue",
            "description": "# Description\n\nTest\n\n# Acceptance Criteria\n\n- AC1",
        }

        # Should not raise and should not call subprocess
        creator.create_issue(issue_config, epic_id=None)

        mock_run.assert_not_called()
