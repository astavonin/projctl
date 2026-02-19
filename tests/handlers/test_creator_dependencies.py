"""Tests for dependency handling in ci_platform_manager.handlers.creator module."""

import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from ci_platform_manager.config import Config
from ci_platform_manager.exceptions import PlatformError
from ci_platform_manager.handlers.creator import EpicIssueCreator


class TestParseDependencyReference:
    """Test _parse_dependency_reference method."""

    def test_parse_yaml_local_id(self, new_config_path: Path) -> None:
        """Parse YAML-local ID string."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        identifier, is_external = creator._parse_dependency_reference("design-task")

        assert identifier == "design-task"
        assert is_external is False

    def test_parse_external_integer(self, new_config_path: Path) -> None:
        """Parse external GitLab IID as integer."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        identifier, is_external = creator._parse_dependency_reference(13)

        assert identifier == "13"
        assert is_external is True

    def test_parse_external_string_with_hash(self, new_config_path: Path) -> None:
        """Parse external GitLab IID as string with # prefix."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        identifier, is_external = creator._parse_dependency_reference("#42")

        assert identifier == "42"
        assert is_external is True

    def test_parse_numeric_string_is_yaml_local(self, new_config_path: Path) -> None:
        """Numeric string without # is treated as YAML-local ID."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        identifier, is_external = creator._parse_dependency_reference("123")

        assert identifier == "123"
        assert is_external is False

    def test_parse_invalid_negative_iid(self, new_config_path: Path) -> None:
        """Negative integer IID raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        with pytest.raises(ValueError, match="must be positive integer"):
            creator._parse_dependency_reference(-1)

    def test_parse_invalid_zero_iid(self, new_config_path: Path) -> None:
        """Zero IID raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        with pytest.raises(ValueError, match="must be positive integer"):
            creator._parse_dependency_reference(0)

    def test_parse_invalid_zero_with_hash(self, new_config_path: Path) -> None:
        """Test that #0 is rejected as invalid."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        with pytest.raises(ValueError, match="Invalid external IID format: #0"):
            creator._parse_dependency_reference("#0")

    def test_parse_invalid_empty_hash(self, new_config_path: Path) -> None:
        """Empty string after # raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        with pytest.raises(ValueError, match="Invalid external IID format"):
            creator._parse_dependency_reference("#")

    def test_parse_invalid_non_numeric_hash(self, new_config_path: Path) -> None:
        """Non-numeric string after # raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        with pytest.raises(ValueError, match="Invalid external IID format"):
            creator._parse_dependency_reference("#abc")

    def test_parse_invalid_negative_hash(self, new_config_path: Path) -> None:
        """Negative number after # raises ValueError."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        with pytest.raises(ValueError, match="Invalid external IID format"):
            creator._parse_dependency_reference("#-5")

    def test_parse_yaml_local_with_special_chars(self, new_config_path: Path) -> None:
        """YAML-local IDs can contain special characters."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        identifier, is_external = creator._parse_dependency_reference("task-123-alpha")

        assert identifier == "task-123-alpha"
        assert is_external is False


class TestGetProjectIdForValidation:
    """Test project ID extraction for validation."""

    @patch("ci_platform_manager.utils.git_helpers.subprocess.run")
    def test_get_project_id_gitlab_url(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Test extraction from GitLab URL."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock git remote get-url origin returning GitLab URL
        mock_run.return_value = Mock(
            stdout="https://gitlab.com/namespace/project.git\n",
            stderr="",
            returncode=0
        )

        project_id = creator._get_project_id_for_validation()

        assert project_id == "namespace/project"
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["git", "remote", "get-url", "origin"]

    @patch("ci_platform_manager.utils.git_helpers.subprocess.run")
    def test_get_project_id_github_url(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Test extraction from GitHub URL."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock git remote get-url origin returning GitHub URL
        mock_run.return_value = Mock(
            stdout="https://github.com/namespace/project.git\n",
            stderr="",
            returncode=0
        )

        project_id = creator._get_project_id_for_validation()

        assert project_id == "namespace/project"

    @patch("ci_platform_manager.utils.git_helpers.subprocess.run")
    def test_get_project_id_with_trailing_slash(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Test extraction with trailing slash."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock git remote returning URL without .git suffix
        mock_run.return_value = Mock(
            stdout="https://gitlab.com/namespace/project\n",
            stderr="",
            returncode=0
        )

        project_id = creator._get_project_id_for_validation()

        assert project_id == "namespace/project"

    @patch("ci_platform_manager.utils.git_helpers.subprocess.run")
    def test_get_project_id_ssh_format(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Test extraction from SSH format URL."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock git remote returning SSH format
        mock_run.return_value = Mock(
            stdout="git@gitlab.com:namespace/project.git\n",
            stderr="",
            returncode=0
        )

        project_id = creator._get_project_id_for_validation()

        assert project_id == "namespace/project"

    @patch("ci_platform_manager.utils.git_helpers.subprocess.run")
    def test_get_project_id_not_in_git_repo(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Test when not in a git repository."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock git command failure
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ["git", "remote"], stderr="not a git repository"
        )

        project_id = creator._get_project_id_for_validation()

        assert project_id is None


class TestValidateExternalIssueExists:
    """Test _validate_external_issue_exists method."""

    @patch("subprocess.run")
    def test_validate_external_issue_exists_success(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """External issue validation succeeds when issue exists."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock successful API call
        mock_run.return_value = Mock(
            stdout='{"iid": 13, "title": "Test Issue"}', stderr="", returncode=0
        )

        result = creator._validate_external_issue_exists("13", "group/project")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "api" in call_args
        assert "projects/group%2Fproject/issues/13" in call_args

    @patch("subprocess.run")
    def test_validate_external_issue_not_found(self, mock_run: Mock, new_config_path: Path) -> None:
        """External issue validation fails gracefully when issue not found."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock API call failure (404)
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["glab", "api"], stderr="404 Not Found"
        )

        result = creator._validate_external_issue_exists("999", "group/project")

        assert result is False


class TestValidateExternalDependencies:
    """Test _validate_external_dependencies method."""

    @patch("subprocess.run")
    def test_validate_external_dependencies_all_valid(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """All external dependencies are valid."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock successful API calls
        mock_run.return_value = Mock(stdout='{"iid": 13, "title": "Test"}', stderr="", returncode=0)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": [13, "#42"]},
            {"id": "task-2", "title": "Task 2", "dependencies": ["task-1"]},
        ]

        errors = creator._validate_external_dependencies(issues, "group/project")

        assert errors == []
        # Should call API twice (once for 13, once for 42)
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_validate_external_dependencies_some_invalid(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Some external dependencies are invalid."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock API: first call succeeds, second fails
        mock_run.side_effect = [
            Mock(stdout='{"iid": 13}', stderr="", returncode=0),
            subprocess.CalledProcessError(1, ["glab", "api"], stderr="404 Not Found"),
        ]

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": [13, "#999"]},
        ]

        errors = creator._validate_external_dependencies(issues, "group/project")

        assert len(errors) == 1
        assert "External issue #999" in errors[0]
        assert "not found" in errors[0]

    @patch("subprocess.run")
    def test_validate_external_dependencies_mixed(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Mixed YAML-local and external dependencies."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock successful API call for external dependency
        mock_run.return_value = Mock(stdout='{"iid": 13}', stderr="", returncode=0)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": []},
            {"id": "task-2", "title": "Task 2", "dependencies": ["task-1", 13]},
        ]

        errors = creator._validate_external_dependencies(issues, "group/project")

        assert errors == []
        # Should only call API once for external dependency
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_validate_external_dependencies_cache(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Same external IID validated only once (caching)."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Mock successful API call
        mock_run.return_value = Mock(stdout='{"iid": 13}', stderr="", returncode=0)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": [13]},
            {"id": "task-2", "title": "Task 2", "dependencies": [13, "#13"]},
            {"id": "task-3", "title": "Task 3", "dependencies": [13]},
        ]

        errors = creator._validate_external_dependencies(issues, "group/project")

        assert errors == []
        # Should only call API once despite multiple references to same IID
        assert mock_run.call_count == 1

    def test_validate_external_dependencies_invalid_format(self, new_config_path: Path) -> None:
        """Invalid dependency format is caught."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": [-1]},
        ]

        errors = creator._validate_external_dependencies(issues, "group/project")

        assert len(errors) == 1
        assert "Invalid dependency reference" in errors[0]
        assert "must be positive integer" in errors[0]

    @patch("subprocess.run")
    def test_validate_external_dependencies_no_external(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """No external dependencies skips validation."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": []},
            {"id": "task-2", "title": "Task 2", "dependencies": ["task-1"]},
        ]

        errors = creator._validate_external_dependencies(issues, "group/project")

        assert errors == []
        # Should not call API
        mock_run.assert_not_called()


class TestCreateDependencyLinks:
    """Test _create_dependency_links method with external dependencies."""

    @patch("subprocess.run")
    def test_create_dependency_links_external_only(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Create dependency links with external dependencies only."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Setup: simulate created issues
        creator.created_issues = [
            {"id": "https://gitlab.com/group/project/-/issues/100", "title": "Task 1"}
        ]
        creator.issue_id_mapping = {
            "task-1": {"iid": "100", "url": "https://gitlab.com/group/project/-/issues/100"}
        }

        # Mock API calls for dependency linking
        mock_run.return_value = Mock(stdout='{"id": 1}', stderr="", returncode=0)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": [13, "#42"]},
        ]

        creator._create_dependency_links(issues)

        # Should create 2 dependency links
        assert mock_run.call_count == 2
        call_args_list = [call[0][0] for call in mock_run.call_args_list]

        # Verify both dependencies were linked
        assert any("issues/13/links" in " ".join(args) for args in call_args_list)
        assert any("issues/42/links" in " ".join(args) for args in call_args_list)

    @patch("subprocess.run")
    def test_create_dependency_links_mixed(self, mock_run: Mock, new_config_path: Path) -> None:
        """Create dependency links with mixed YAML-local and external dependencies."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Setup: simulate created issues
        creator.created_issues = [
            {"id": "https://gitlab.com/group/project/-/issues/100", "title": "Task 1"},
            {"id": "https://gitlab.com/group/project/-/issues/101", "title": "Task 2"},
        ]
        creator.issue_id_mapping = {
            "task-1": {"iid": "100", "url": "https://gitlab.com/group/project/-/issues/100"},
            "task-2": {"iid": "101", "url": "https://gitlab.com/group/project/-/issues/101"},
        }

        # Mock API calls
        mock_run.return_value = Mock(stdout='{"id": 1}', stderr="", returncode=0)

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": []},
            {"id": "task-2", "title": "Task 2", "dependencies": ["task-1", 13]},
        ]

        creator._create_dependency_links(issues)

        # Should create 2 dependency links (task-1 blocks task-2, and #13 blocks task-2)
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_create_dependency_links_invalid_iid_graceful(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """Invalid IID is handled gracefully without failing entire operation."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Setup
        creator.created_issues = [
            {"id": "https://gitlab.com/group/project/-/issues/100", "title": "Task 1"}
        ]
        creator.issue_id_mapping = {
            "task-1": {"iid": "100", "url": "https://gitlab.com/group/project/-/issues/100"}
        }

        # Mock API: first call fails, method should continue
        mock_run.return_value = Mock(stdout='{"id": 1}', stderr="", returncode=0)

        issues = [
            # Invalid dependency will be caught by _parse_dependency_reference
            {"id": "task-1", "title": "Task 1", "dependencies": ["valid-yaml-id", 42]},
        ]

        # Should not raise, just log warnings
        creator._create_dependency_links(issues)

        # Should attempt to create link for 42 (valid-yaml-id won't be found in mapping)
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_create_dependency_links_yaml_local_not_found(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        """YAML-local dependency not found is handled gracefully."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Setup
        creator.created_issues = [
            {"id": "https://gitlab.com/group/project/-/issues/100", "title": "Task 1"}
        ]
        creator.issue_id_mapping = {
            "task-1": {"iid": "100", "url": "https://gitlab.com/group/project/-/issues/100"}
        }

        issues = [
            {"id": "task-1", "title": "Task 1", "dependencies": ["nonexistent-task"]},
        ]

        # Should not raise, just log warning
        creator._create_dependency_links(issues)

        # Should not attempt to create any links
        mock_run.assert_not_called()


class TestIntegrationWithYAML:
    """Integration tests with full YAML file processing."""

    @patch("subprocess.run")
    def test_process_yaml_with_external_dependencies(
        self, mock_run: Mock, new_config_path: Path, temp_dir: Path
    ) -> None:
        """Process YAML file with external dependencies."""
        config = Config(new_config_path)
        creator = EpicIssueCreator(config)

        # Create test YAML file
        yaml_content = """
epic:
  id: 21

issues:
  - id: "task-1"
    title: "Task 1"
    description: |
      # Description
      Task description

      # Acceptance Criteria
      - AC1
    dependencies: [13, "#42"]

  - id: "task-2"
    title: "Task 2"
    description: |
      # Description
      Another task

      # Acceptance Criteria
      - AC1
    dependencies: ["task-1"]
"""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text(yaml_content)

        # Mock responses
        def mock_run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd)

            # Mock project validation
            if "repo view --json nameWithOwner" in cmd_str:
                return Mock(stdout='{"nameWithOwner": "group/project"}', returncode=0)

            # Mock external issue validation
            if "api" in cmd_str and "issues/13" in cmd_str:
                return Mock(stdout='{"iid": 13}', returncode=0)
            if "api" in cmd_str and "issues/42" in cmd_str:
                return Mock(stdout='{"iid": 42}', returncode=0)

            # Mock issue creation
            if "issue create" in cmd_str:
                if "Task 1" in cmd_str:
                    return Mock(
                        stdout="https://gitlab.com/group/project/-/issues/100", returncode=0
                    )
                elif "Task 2" in cmd_str:
                    return Mock(
                        stdout="https://gitlab.com/group/project/-/issues/101", returncode=0
                    )

            # Mock dependency link creation
            if "links" in cmd_str:
                return Mock(stdout='{"id": 1}', returncode=0)

            return Mock(stdout="", returncode=0)

        mock_run.side_effect = mock_run_side_effect

        # Process YAML
        creator.process_yaml_file(yaml_file)

        # Verify issues were created
        assert len(creator.created_issues) == 2

        # Verify dependency links were attempted (2 for task-1, 1 for task-2)
        api_calls = [
            call[0][0] for call in mock_run.call_args_list if "links" in " ".join(call[0][0])
        ]
        assert len(api_calls) == 3
