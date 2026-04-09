"""Tests for projctl.utils.mr_builder module."""

import types

import pytest

from projctl.utils.mr_builder import append_common_mr_flags


def _args(**kwargs) -> types.SimpleNamespace:
    """Build a fake args namespace with all MR flag fields defaulted to None/empty."""
    defaults = {
        "assignee": [],
        "reviewer": [],
        "label": [],
        "milestone": None,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


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
