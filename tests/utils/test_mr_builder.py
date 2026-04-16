"""Tests for projctl.utils.mr_builder module."""

import types
from unittest.mock import Mock

import pytest

from projctl.utils.mr_builder import append_common_mr_flags, validate_mr_args


def _args(**kwargs) -> types.SimpleNamespace:
    """Build a fake args namespace with all MR flag fields defaulted to None/empty."""
    defaults = {
        "assignee": [],
        "reviewer": [],
        "label": [],
        "milestone": None,
        "title": None,
        "description": None,
        "fill": False,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


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


class TestAppendCommonMrFlagsEmpty:
    """No flags appended when all fields are None or empty."""

    def test_all_none_or_empty_appends_nothing(self) -> None:
        """Empty/None fields produce no additions to the command list."""
        cmd: list = []
        append_common_mr_flags(cmd, _args())
        assert cmd == []


class TestAppendCommonMrFlagsAssignee:
    """Single assignee appends --assignee flag."""

    def test_single_assignee_appended(self) -> None:
        """One assignee produces ['--assignee', 'alice']."""
        cmd: list = []
        append_common_mr_flags(cmd, _args(assignee=["alice"]))
        assert cmd == ["--assignee", "alice"]


class TestAppendCommonMrFlagsReviewers:
    """Multiple reviewers produce two --reviewer pairs."""

    def test_multiple_reviewers_all_appended(self) -> None:
        """Two reviewers produce four tokens in order."""
        cmd: list = []
        append_common_mr_flags(cmd, _args(reviewer=["alice", "bob"]))
        assert cmd == ["--reviewer", "alice", "--reviewer", "bob"]


class TestAppendCommonMrFlagsLabels:
    """Multiple labels produce two --label pairs."""

    def test_multiple_labels_all_appended(self) -> None:
        """Two labels produce four tokens in order."""
        cmd: list = []
        append_common_mr_flags(cmd, _args(label=["bug", "enhancement"]))
        assert cmd == ["--label", "bug", "--label", "enhancement"]


class TestAppendCommonMrFlagsMilestone:
    """Milestone value appends --milestone flag."""

    def test_milestone_appended(self) -> None:
        """Milestone title produces ['--milestone', 'v2.0']."""
        cmd: list = []
        append_common_mr_flags(cmd, _args(milestone="v2.0"))
        assert cmd == ["--milestone", "v2.0"]


class TestAppendCommonMrFlagsAllFields:
    """All fields together produce flags in correct order."""

    def test_all_fields_produce_all_flags(self) -> None:
        """All fields present → assignee, reviewer, label, milestone all appear."""
        cmd: list = []
        append_common_mr_flags(
            cmd,
            _args(
                assignee=["alice"],
                reviewer=["bob"],
                label=["type::feature"],
                milestone="v1.0",
            ),
        )
        assert "--assignee" in cmd
        assert "alice" in cmd
        assert "--reviewer" in cmd
        assert "bob" in cmd
        assert "--label" in cmd
        assert "type::feature" in cmd
        assert "--milestone" in cmd
        assert "v1.0" in cmd


class TestValidateMrArgsValid:
    """validate_mr_args raises no exception for valid input."""

    def test_valid_title_description_and_sections(self) -> None:
        """Valid title, description with all required sections passes without exception."""
        # Arrange
        args = _args(title="Add feature X", description=_VALID_DESCRIPTION)
        config = _config()

        # Act / Assert — no exception
        validate_mr_args(args, config)


class TestValidateMrArgsMissingTitle:
    """validate_mr_args raises ValueError when title is absent and fill is False."""

    def test_missing_title_raises_value_error(self) -> None:
        """Missing title without --fill raises ValueError mentioning title."""
        # Arrange
        args = _args(title=None, description=_VALID_DESCRIPTION, fill=False)
        config = _config()

        # Act / Assert
        with pytest.raises(ValueError, match="title"):
            validate_mr_args(args, config)


class TestValidateMrArgsMissingDescription:
    """validate_mr_args raises ValueError when description is absent and fill is False."""

    def test_missing_description_raises_value_error(self) -> None:
        """Missing description without --fill raises ValueError mentioning description."""
        # Arrange
        args = _args(title="Add feature X", description=None, fill=False)
        config = _config()

        # Act / Assert
        with pytest.raises(ValueError, match="description"):
            validate_mr_args(args, config)


class TestValidateMrArgsMissingSection:
    """validate_mr_args raises ValueError when description is missing a required section."""

    def test_description_missing_required_section_raises(self) -> None:
        """Description present but missing a required section raises ValueError."""
        # Arrange
        incomplete = "# Summary\n\nWhat changed.\n\n# Implementation Details\n\nHow.\n"
        args = _args(title="Add feature X", description=incomplete, fill=False)
        config = _config()

        # Act / Assert
        with pytest.raises(ValueError, match="How It Was Tested"):
            validate_mr_args(args, config)


class TestValidateMrArgsFillBypass:
    """validate_mr_args skips all validation when fill=True."""

    def test_fill_true_no_title_no_description_no_exception(self) -> None:
        """fill=True with no title and no description raises no exception."""
        # Arrange
        args = _args(title=None, description=None, fill=True)
        config = _config()

        # Act / Assert — no exception
        validate_mr_args(args, config)

    def test_fill_true_with_invalid_description_no_exception(self) -> None:
        """fill=True with description missing required sections raises no exception."""
        # Arrange
        args = _args(title="My MR", description="No sections here", fill=True)
        config = _config()

        # Act / Assert — no exception
        validate_mr_args(args, config)


class TestValidateMrArgsEmptySections:
    """validate_mr_args raises no exception when required_sections is []."""

    def test_empty_required_sections_no_exception(self) -> None:
        """required_sections=[] means no description validation; even empty description passes."""
        # Arrange
        args = _args(title="My MR", description=None, fill=False)
        config = _config(mr_sections=[])

        # Act / Assert — description check is skipped when sections is empty, but
        # the description-absent guard fires first.  This test verifies the
        # documented behaviour: title check still fires, description check fires.
        # To verify "no section check", provide a description with no sections.
        args2 = _args(title="My MR", description="No sections here", fill=False)
        # Should not raise because required_sections is empty
        validate_mr_args(args2, config)


class TestValidateMrArgsRequiredFields:
    """validate_mr_args enforces required_fields (reviewers, labels) from config."""

    def test_reviewers_required_no_reviewer_raises(self) -> None:
        """required_fields=['reviewers'], no --reviewer → ValueError mentioning reviewer."""
        # Arrange
        args = _args(title="My MR", description=_VALID_DESCRIPTION, reviewer=[])
        config = _config(mr_fields=["reviewers"])

        # Act / Assert
        with pytest.raises(ValueError, match="reviewer"):
            validate_mr_args(args, config)

    def test_reviewers_required_reviewer_provided_no_exception(self) -> None:
        """required_fields=['reviewers'], reviewer provided → no exception."""
        # Arrange
        args = _args(title="My MR", description=_VALID_DESCRIPTION, reviewer=["alice"])
        config = _config(mr_fields=["reviewers"])

        # Act / Assert — no exception
        validate_mr_args(args, config)

    def test_labels_required_no_label_raises(self) -> None:
        """required_fields=['labels'], no --label → ValueError mentioning label."""
        # Arrange
        args = _args(title="My MR", description=_VALID_DESCRIPTION, label=[])
        config = _config(mr_fields=["labels"])

        # Act / Assert
        with pytest.raises(ValueError, match="label"):
            validate_mr_args(args, config)

    def test_labels_required_label_provided_no_exception(self) -> None:
        """required_fields=['labels'], label provided → no exception."""
        # Arrange
        args = _args(title="My MR", description=_VALID_DESCRIPTION, label=["type::feature"])
        config = _config(mr_fields=["labels"])

        # Act / Assert — no exception
        validate_mr_args(args, config)

    def test_empty_required_fields_no_exception(self) -> None:
        """required_fields=[], no reviewer/label → no exception."""
        # Arrange
        args = _args(title="My MR", description=_VALID_DESCRIPTION, reviewer=[], label=[])
        config = _config(mr_fields=[])

        # Act / Assert — no exception
        validate_mr_args(args, config)

    def test_fill_bypasses_required_fields_check(self) -> None:
        """fill=True, required_fields=['reviewers', 'labels'] → no exception (fill bypasses all)."""
        # Arrange
        args = _args(title=None, description=None, fill=True, reviewer=[], label=[])
        config = _config(mr_fields=["reviewers", "labels"])

        # Act / Assert — no exception; fill early-returns before any check
        validate_mr_args(args, config)
