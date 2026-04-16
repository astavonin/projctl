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
        "fill": False,
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


def _config(mr_sections=None, mr_fields=None) -> Mock:
    """Build a mock Config that returns the given MR required sections and required fields."""
    if mr_sections is None:
        mr_sections = ["Summary", "Implementation Details", "How It Was Tested"]
    if mr_fields is None:
        mr_fields = []
    mock = Mock()
    mock.get_required_mr_sections.return_value = mr_sections
    mock.get_required_mr_fields.return_value = mr_fields
    return mock


_VALID_DESCRIPTION = (
    "# Summary\n\nWhat changed.\n\n"
    "# Implementation Details\n\nHow it was done.\n\n"
    "# How It Was Tested\n\nTest approach.\n"
)


class TestCreatePrMinimal:
    """Minimal invocation constructs a valid gh pr create command."""

    @patch("subprocess.run")
    def test_create_pr_minimal(self, mock_run: Mock) -> None:
        """Title-only PR creates correct command."""
        mock_run.return_value = Mock(
            stdout="https://github.com/owner/repo/pull/1", stderr="", returncode=0
        )

        args = _args(title="Add feature X", description=_VALID_DESCRIPTION)
        result = cmd_create_pr(args, _config())

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
            description=_VALID_DESCRIPTION,
            draft=True,
            assignee=["octocat"],
            reviewer=["reviewer1"],
            label=["enhancement"],
            milestone="v1.0",
            target_branch="main",
        )
        result = cmd_create_pr(args, _config())

        assert result == 0
        cmd = mock_run.call_args[0][0]
        assert "--title" in cmd
        assert "Full PR" in cmd
        assert "--body" in cmd
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
        args = _args(title="Dry PR", description=_VALID_DESCRIPTION, dry_run=True)
        result = cmd_create_pr(args, _config())

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
        args = _args(title="Error PR", description=_VALID_DESCRIPTION)
        result = cmd_create_pr(args, _config())

        assert result == 1

    @patch("subprocess.run")
    def test_create_pr_returns_1_on_os_error(self, mock_run: Mock) -> None:
        """OSError (gh not installed) returns exit code 1."""
        mock_run.side_effect = FileNotFoundError("gh not found")
        args = _args(title="Error PR", description=_VALID_DESCRIPTION)
        result = cmd_create_pr(args, _config())

        assert result == 1


class TestCreatePrValidation:
    """Validation tests for cmd_create_pr."""

    @patch("subprocess.run")
    def test_missing_title_returns_1_subprocess_not_called(self, mock_run: Mock) -> None:
        """Missing title without --fill returns 1; subprocess never called."""
        # Arrange
        args = _args(title=None, description=_VALID_DESCRIPTION, fill=False)

        # Act
        result = cmd_create_pr(args, _config())

        # Assert
        assert result == 1
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_missing_description_returns_1_subprocess_not_called(self, mock_run: Mock) -> None:
        """Missing description without --fill returns 1; subprocess never called."""
        # Arrange
        args = _args(title="Add feature X", description=None, fill=False)

        # Act
        result = cmd_create_pr(args, _config())

        # Assert
        assert result == 1
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_description_missing_section_returns_1(self, mock_run: Mock) -> None:
        """Description missing required section returns 1; subprocess never called."""
        # Arrange
        incomplete = "# Summary\n\nWhat changed.\n\n# Implementation Details\n\nHow.\n"
        args = _args(title="Add feature X", description=incomplete, fill=False)

        # Act
        result = cmd_create_pr(args, _config())

        # Assert
        assert result == 1
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_fill_true_returns_0_subprocess_called(self, mock_run: Mock) -> None:
        """fill=True bypasses validation and calls subprocess."""
        # Arrange
        mock_run.return_value = Mock(
            stdout="https://github.com/owner/repo/pull/44", stderr="", returncode=0
        )
        args = _args(fill=True)

        # Act
        result = cmd_create_pr(args, _config())

        # Assert
        assert result == 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_dry_run_with_invalid_description_returns_1(self, mock_run: Mock) -> None:
        """Dry-run with invalid description returns 1; subprocess never called."""
        # Arrange
        args = _args(title="My PR", description="No sections here", dry_run=True, fill=False)

        # Act
        result = cmd_create_pr(args, _config())

        # Assert
        assert result == 1
        mock_run.assert_not_called()


class TestCreatePrRequiredFields:
    """required_fields validation fires on GitHub PR path via shared validate_mr_args."""

    @patch("subprocess.run")
    def test_reviewers_required_no_reviewer_returns_1_subprocess_not_called(
        self, mock_run: Mock
    ) -> None:
        """required_fields=['reviewers'], no --reviewer → cmd_create_pr returns 1, no subprocess."""
        # Arrange
        args = _args(title="My PR", description=_VALID_DESCRIPTION, reviewer=[], fill=False)

        # Act
        result = cmd_create_pr(args, _config(mr_fields=["reviewers"]))

        # Assert
        assert result == 1
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_labels_required_no_label_returns_1_subprocess_not_called(
        self, mock_run: Mock
    ) -> None:
        """required_fields=['labels'], no --label → cmd_create_pr returns 1, no subprocess."""
        # Arrange
        args = _args(
            title="My PR",
            description=(
                "# Summary\n\n---\n# Implementation Details\n\n---\n# How It Was Tested\n"
            ),
        )
        config = _config()
        config.get_required_mr_fields.return_value = ["labels"]
        config.get_required_mr_sections.return_value = []

        # Act
        result = cmd_create_pr(args, config)

        # Assert
        assert result == 1
        mock_run.assert_not_called()
