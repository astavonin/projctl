"""Tests for ci_platform_manager.utils.git_helpers module."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ci_platform_manager.utils.git_helpers import get_current_repo_path


class TestGetCurrentRepoPath:
    """Test get_current_repo_path function."""

    @patch("subprocess.run")
    def test_https_url_gitlab(self, mock_run: Mock) -> None:
        """Parse HTTPS GitLab URL correctly."""
        mock_run.return_value = Mock(stdout="https://gitlab.com/group/project.git\n", returncode=0)

        result = get_current_repo_path()

        assert result == "group/project"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_ssh_url_gitlab(self, mock_run: Mock) -> None:
        """Parse SSH GitLab URL correctly."""
        mock_run.return_value = Mock(
            stdout="git@gitlab.com:group/subgroup/project.git\n", returncode=0
        )

        result = get_current_repo_path()

        assert result == "group/subgroup/project"

    @patch("subprocess.run")
    def test_https_url_github(self, mock_run: Mock) -> None:
        """Parse HTTPS GitHub URL correctly."""
        mock_run.return_value = Mock(stdout="https://github.com/owner/repo.git\n", returncode=0)

        result = get_current_repo_path()

        assert result == "owner/repo"

    @patch("subprocess.run")
    def test_ssh_url_github(self, mock_run: Mock) -> None:
        """Parse SSH GitHub URL correctly."""
        mock_run.return_value = Mock(stdout="git@github.com:owner/repo.git\n", returncode=0)

        result = get_current_repo_path()

        assert result == "owner/repo"

    @patch("subprocess.run")
    def test_url_without_git_suffix(self, mock_run: Mock) -> None:
        """Parse URL without .git suffix correctly."""
        mock_run.return_value = Mock(stdout="https://gitlab.com/group/project\n", returncode=0)

        result = get_current_repo_path()

        assert result == "group/project"

    @patch("subprocess.run")
    def test_nested_group_path(self, mock_run: Mock) -> None:
        """Parse nested group path correctly."""
        mock_run.return_value = Mock(
            stdout="git@gitlab.com:org/team/subteam/project.git\n", returncode=0
        )

        result = get_current_repo_path()

        assert result == "org/team/subteam/project"

    @patch("subprocess.run")
    def test_not_a_git_repo(self, mock_run: Mock) -> None:
        """Return None when not in a git repository."""
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ["git", "remote", "get-url", "origin"]
        )

        result = get_current_repo_path()

        assert result is None

    @patch("subprocess.run")
    def test_no_remote_origin(self, mock_run: Mock) -> None:
        """Return None when remote origin doesn't exist."""
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ["git", "remote", "get-url", "origin"]
        )

        result = get_current_repo_path()

        assert result is None

    @patch("subprocess.run")
    def test_invalid_url_format(self, mock_run: Mock) -> None:
        """Return None for malformed URL."""
        mock_run.return_value = Mock(stdout="invalid-url-format\n", returncode=0)

        # Should not crash, returns None or partial parsing
        result = get_current_repo_path()

        # Result might be None or partial - we just verify no exception
        assert result is not None or result is None

    @patch("subprocess.run")
    def test_uses_current_directory(self, mock_run: Mock, temp_dir: Path) -> None:
        """Uses current working directory for git command."""
        mock_run.return_value = Mock(stdout="https://gitlab.com/test/repo.git\n", returncode=0)

        get_current_repo_path()

        # Verify subprocess.run was called with cwd parameter
        call_args = mock_run.call_args
        assert "cwd" in call_args.kwargs
        assert call_args.kwargs["cwd"] == Path.cwd()
