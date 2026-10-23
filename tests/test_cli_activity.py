"""Tests for the 'activity' subcommand (projctl.handlers.activity, projctl.cli.cmd_activity).

Purely local git, so every test here mocks subprocess.run directly with a
fixed transcript — never a real repository — matching the module's own
"no network, no GitLab API" contract.
"""

import json
import os
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional
from unittest.mock import patch

import pytest

from projctl.cli import cmd_activity, main
from projctl.exceptions import PlatformError
from projctl.handlers.activity import ActivityHandler, ActivityReport, IssueActivity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# UTC+6, matching the offset the design doc's own local-day reasoning is
# pinned to ("this host runs at UTC+6").
_FIXED_TZ = timezone(timedelta(hours=6))
_TARGET_DAY = date(2026, 8, 13)


def _completed(
    cmd: List[str], returncode: int, stdout: str, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=cmd, returncode=returncode, stdout=stdout, stderr=stderr
    )


def _git_stub(
    *,
    toplevel: str = "/repo",
    toplevel_returncode: int = 0,
    reflog: str = "",
    status: str = "",
    branch: str = "",
    calls: Optional[List[List[str]]] = None,
) -> Callable[..., subprocess.CompletedProcess]:
    """Build a subprocess.run replacement keyed by git subcommand.

    Dispatching by subcommand name (rather than an ordered side_effect
    list) means each test only has to supply the transcripts relevant to
    the evidence path it exercises, and stays correct even if report()'s
    internal call order changes. `calls`, when given, records every
    invoked argv so a test can assert a subcommand was (or was not)
    consulted — e.g. 'branch' must never be called when no file was
    modified on the date.
    """

    def _run(cmd: List[str], **_kwargs) -> subprocess.CompletedProcess:
        assert cmd[0] == "git"
        if calls is not None:
            calls.append(cmd)
        sub = cmd[1]
        if sub == "rev-parse":
            return _completed(cmd, toplevel_returncode, toplevel)
        if sub == "reflog":
            return _completed(cmd, 0, reflog)
        if sub == "status":
            return _completed(cmd, 0, status)
        if sub == "branch":
            return _completed(cmd, 0, branch)
        raise AssertionError(f"unexpected git subcommand invoked: {cmd}")

    return _run


def _reflog_line(sha: str, iso_dt: str, subject: str) -> str:
    return f"{sha} HEAD@{{{iso_dt}}}: {subject}"


def _set_mtime(path: Path, day: date, tz=_FIXED_TZ) -> None:
    """Set a file's mtime to local noon on `day`, in `tz`."""
    ts = datetime.combine(day, time(12, 0), tzinfo=tz).timestamp()
    os.utime(path, (ts, ts))


# ---------------------------------------------------------------------------
# _parse_status_entries — pure parsing, no subprocess involved
# ---------------------------------------------------------------------------


class TestParseStatusEntries:
    """Parsing of `git status --porcelain -z` output."""

    def test_untracked_file(self) -> None:
        output = "?? file.txt\0"
        assert ActivityHandler._parse_status_entries(output) == ["file.txt"]

    def test_path_with_space(self) -> None:
        output = "?? un tracked.txt\0"
        assert ActivityHandler._parse_status_entries(output) == ["un tracked.txt"]

    def test_rename_entry_yields_only_the_new_path(self) -> None:
        """A rename carries an extra NUL field (the pre-rename path) that must be
        consumed and discarded, not treated as a second entry."""
        output = "RM new name.txt\0old name.txt\0"
        assert ActivityHandler._parse_status_entries(output) == ["new name.txt"]

    def test_copy_entry_also_consumes_the_original_path_field(self) -> None:
        output = "C  copied.txt\0source.txt\0"
        assert ActivityHandler._parse_status_entries(output) == ["copied.txt"]

    def test_modified_and_rename_and_untracked_together(self) -> None:
        output = "RM new name.txt\0old name.txt\0M  changed.txt\0?? untracked.txt\0"
        assert ActivityHandler._parse_status_entries(output) == [
            "new name.txt",
            "changed.txt",
            "untracked.txt",
        ]

    def test_empty_output_yields_no_entries(self) -> None:
        assert ActivityHandler._parse_status_entries("") == []


# ---------------------------------------------------------------------------
# Reflog evidence
# ---------------------------------------------------------------------------


class TestScanReflog:
    """ActivityHandler._scan_reflog / report()'s reflog-sourced issue attribution."""

    def test_entry_with_ref_is_attributed(self) -> None:
        reflog = _reflog_line("abc1234", "2026-08-13T09:00:00+06:00", "commit: Fix thing. Ref #10")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(10, "reflog", 1)]
        assert report.unattributed_events == 0

    def test_entry_without_ref_is_counted_as_unattributed_not_dropped(self) -> None:
        reflog = _reflog_line("abc1234", "2026-08-13T09:00:00+06:00", "commit: Fix thing")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog, status="")):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_events == 1

    def test_commit_and_amend_subjects_both_count(self) -> None:
        reflog = "\n".join(
            [
                _reflog_line("aaa1111", "2026-08-13T09:00:00+06:00", "commit: work. Ref #10"),
                _reflog_line(
                    "bbb2222", "2026-08-13T10:00:00+06:00", "commit (amend): more work. Ref #10"
                ),
            ]
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(10, "reflog", 2)]

    def test_initial_commit_subject_is_not_evidence(self) -> None:
        """'commit (initial):' is a real reflog subject for a repo's first commit,
        but the spec enumerates only 'commit:' and 'commit (amend):' as evidence."""
        reflog = _reflog_line(
            "aaa1111", "2026-08-13T09:00:00+06:00", "commit (initial): init. Ref #10"
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog, status="")):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_events == 0

    def test_non_commit_subjects_are_ignored_entirely(self) -> None:
        """checkout/merge/etc. reflog entries are neither attributed nor counted
        as unattributed — they carry no commit evidence at all."""
        reflog = "\n".join(
            [
                _reflog_line(
                    "aaa1111", "2026-08-13T09:00:00+06:00", "checkout: moving from a to b"
                ),
                _reflog_line("bbb2222", "2026-08-13T09:05:00+06:00", "pull: Fast-forward"),
            ]
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog, status="")):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_events == 0

    def test_entry_at_end_of_local_day_boundary_is_included(self) -> None:
        reflog = _reflog_line("aaa1111", "2026-08-13T23:59:59+06:00", "commit: late work. Ref #10")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(10, "reflog", 1)]

    def test_entry_just_past_local_midnight_falls_to_the_next_day(self) -> None:
        reflog = _reflog_line(
            "aaa1111", "2026-08-14T00:00:00+06:00", "commit: past midnight. Ref #10"
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog, status="")):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_events == 0

    def test_entry_recorded_in_a_different_offset_is_converted_to_local_day(self) -> None:
        """23:30 UTC on 2026-08-13 is 05:30 local (+06:00) on 2026-08-14 — a naive
        string comparison against the recorded date would misclassify this."""
        reflog = _reflog_line("aaa1111", "2026-08-13T23:30:00+00:00", "commit: work. Ref #10")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))

        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog, status="")):
            report_13 = handler.report("2026-08-13")
        assert report_13.issues == []

        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report_14 = handler.report("2026-08-14")
        assert report_14.issues == [IssueActivity(10, "reflog", 1)]

    def test_empty_reflog_yields_no_issues_and_falls_through(self) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog="", status="")):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_events == 0

    def test_blank_reflog_line_is_skipped(self) -> None:
        reflog = "\n\n" + _reflog_line(
            "aaa1111", "2026-08-13T09:00:00+06:00", "commit: work. Ref #10"
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(10, "reflog", 1)]

    def test_reflog_entry_with_unparsable_date_is_skipped_not_fatal(self) -> None:
        reflog = _reflog_line("aaa1111", "not-a-timestamp", "commit: work. Ref #10")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog, status="")):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_events == 0

    def test_unparsable_reflog_line_is_skipped_not_fatal(self) -> None:
        reflog = "\n".join(
            [
                "this is not a reflog line",
                _reflog_line("aaa1111", "2026-08-13T09:00:00+06:00", "commit: work. Ref #10"),
            ]
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(10, "reflog", 1)]


# ---------------------------------------------------------------------------
# Working-tree evidence (only reached when the reflog attributes nothing)
# ---------------------------------------------------------------------------


class TestWorkingTreeFallback:
    """report()'s working-tree evidence path, and the branch-consultation gate."""

    def test_file_modified_on_date_vs_weeks_ago(self, tmp_path: Path) -> None:
        recent = tmp_path / "recent.txt"
        old = tmp_path / "old.txt"
        recent.write_text("x")
        old.write_text("y")
        _set_mtime(recent, _TARGET_DAY)
        _set_mtime(old, _TARGET_DAY - timedelta(days=21))

        status = "?? recent.txt\0?? old.txt\0"
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=status, branch="feature/4-vision"),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(4, "branch", 1)]

    def test_path_with_space_is_matched(self, tmp_path: Path) -> None:
        target = tmp_path / "un tracked.txt"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        status = "?? un tracked.txt\0"
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=status, branch="feature/4-vision"),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(4, "branch", 1)]

    def test_rename_checks_the_new_paths_mtime_only(self, tmp_path: Path) -> None:
        new_path = tmp_path / "new name.txt"
        new_path.write_text("x")
        _set_mtime(new_path, _TARGET_DAY)
        # old name.txt deliberately does not exist on disk (it was renamed away).

        status = "RM new name.txt\0old name.txt\0"
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=status, branch="feature/4-vision"),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(4, "branch", 1)]

    def test_untracked_file_counts_as_evidence(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        status = "?? file.txt\0"
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=status, branch="fix/9-bug"),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(9, "branch", 1)]

    def test_deleted_path_with_no_mtime_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        status = "?? gone.txt\0"  # never created on disk
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=status),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_files == 0

    def test_branch_without_issue_number_is_unattributed(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        status = "?? file.txt\0"
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=status, branch="main"),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_files == 1
        assert report.unattributed_branch == "main"

    def test_branch_with_no_slash_yields_no_issue(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status="?? file.txt\0", branch="develop"),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_branch == "develop"

    def test_stale_branch_with_no_modified_files_contributes_nothing(self, tmp_path: Path) -> None:
        """A branch left checked out from unrelated old work must not be attributed
        when nothing on disk was actually touched on the date — and the branch
        must never even be queried in that case (see calls assertion below)."""
        calls: List[List[str]] = []
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status="", calls=calls),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == []
        assert report.unattributed_files == 0
        assert report.unattributed_branch is None
        assert all(cmd[1] != "branch" for cmd in calls)

    def test_reflog_issue_found_means_branch_is_never_consulted(self, tmp_path: Path) -> None:
        calls: List[List[str]] = []
        reflog = _reflog_line("aaa1111", "2026-08-13T09:00:00+06:00", "commit: work. Ref #10")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog=reflog, calls=calls),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == [IssueActivity(10, "reflog", 1)]
        assert all(cmd[1] not in ("status", "branch") for cmd in calls)


# ---------------------------------------------------------------------------
# Planning-tree evidence (issues/<N>-<slug>/ and reviews/MR<N>-review.yaml) —
# no subprocess involved, so these hit the real filesystem under tmp_path.
# ---------------------------------------------------------------------------


class TestScanPlanningTree:
    """ActivityHandler._scan_planning_tree — sources 3 (planning) and 4 (review)."""

    def test_padded_issue_folder_strips_leading_zeros(self, tmp_path: Path) -> None:
        issue_dir = tmp_path / "planning" / "epic" / "milestone-01" / "issues" / "054-yocto-setup"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "analysis.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {54: 1}
        assert mr_counts == {}

    def test_file_nested_several_levels_below_issue_folder_is_counted(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "planning" / "epic" / "issues" / "10-foo" / "sub" / "deeper"
        nested_dir.mkdir(parents=True)
        target = nested_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {10: 1}

    def test_file_under_planning_outside_any_issue_folder_yields_nothing(
        self, tmp_path: Path
    ) -> None:
        misc_dir = tmp_path / "planning" / "epic"
        misc_dir.mkdir(parents=True)
        target = misc_dir / "overview.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_reviews_orphan_directory_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        """Pins the `reviews-orphan` member of `_PLANNING_NON_ISSUE_DIRS`:
        without it, '51-ab-fallback-fix' would match the 2-layer component
        pattern and be read as issue 51."""
        orphan_dir = tmp_path / "planning" / "reviews-orphan" / "51-ab-fallback-fix"
        orphan_dir.mkdir(parents=True)
        target = orphan_dir / "observed-failures.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_reviews_orphan_directory_does_not_match_the_mr_pattern(self, tmp_path: Path) -> None:
        """'reviews-orphan/' (hand-managed reviews for unlinked work, per the
        planning-structure convention) must not be mistaken for 'reviews/' —
        the MR pattern requires the exact directory name, not a prefix of
        it. The fixture places the MR-shaped filename directly inside
        'reviews-orphan/' so a relaxed 'reviews[^/]*/' pattern would actually
        claim it; a filename that isn't MR-shaped at all could never
        exercise this collision regardless of how the directory match is
        written.
        """
        orphan_dir = tmp_path / "planning" / "reviews-orphan"
        orphan_dir.mkdir(parents=True)
        target = orphan_dir / "MR5-review.yaml"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_two_layer_issue_folder_is_counted(self, tmp_path: Path) -> None:
        """planning/<goal>/<N>-<slug>/ is the 2-layer shape. A repo using it
        was reporting no planning evidence at all, so a day of design and
        review work on an issue showed up only as whatever its branch or a
        single commit happened to prove."""
        issue_dir = tmp_path / "planning" / "platform-resilience" / "70-hil-device-versions"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "observed-failures.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {70: 1}
        assert mr_counts == {}

    def test_two_layer_issue_folder_strips_leading_zeros(self, tmp_path: Path) -> None:
        issue_dir = tmp_path / "planning" / "goal" / "054-yocto-setup"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "design.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {54: 1}

    def test_two_layer_files_sum_into_one_count(self, tmp_path: Path) -> None:
        issue_dir = tmp_path / "planning" / "goal" / "70-thing"
        issue_dir.mkdir(parents=True)
        for name in ("design.md", "observed-failures.md", "code-review.md"):
            target = issue_dir / name
            target.write_text("x")
            _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {70: 3}

    @pytest.mark.parametrize(
        "slug, expected",
        [
            ("1-first-issue", {1: 1}),
            ("7-alpha", {7: 1}),
            ("999999-thing", {999999: 1}),
            ("1234567-thing", {}),
            ("20261012-notes", {}),
        ],
    )
    def test_flat_issue_id_digit_cap_bounds_the_component(
        self, tmp_path: Path, slug: str, expected: dict
    ) -> None:
        """The component pattern's `{1,6}` digit cap accepts a single-digit
        issue id and a 6-digit one, rejects a 7-digit one, and incidentally
        is the sole barrier against a compact date folder —
        '20261012-notes' has no hyphen for the date guard to anchor on, so
        only the cap stops it from being read as issue 20261012.
        '1-first-issue' pins the zero floor's adjacent value from above,
        complementing test_zero_prefixed_component_is_not_read_as_an_issue
        below, which pins it from below."""
        issue_dir = tmp_path / "planning" / "goal" / slug
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == expected

    def test_zero_prefixed_component_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        """GitLab iids are 1-based, so '0-triage' is not a real issue folder —
        attributing issue 0 would invent a ticket that cannot exist."""
        issue_dir = tmp_path / "planning" / "goal" / "0-triage"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_zero_id_three_layer_issue_folder_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        """The zero floor is a validity check (GitLab iids are 1-based), not
        a disambiguation guard, so unlike the date guard and digit cap it
        also applies to the 3-layer issues/<N>-<slug>/ shape — '0-triage'
        there is no more a real issue folder than it is in the 2-layer
        shape above."""
        issue_dir = tmp_path / "planning" / "epic" / "milestone-01" / "issues" / "0-triage"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_issue_one_in_a_three_layer_issue_folder_is_counted(self, tmp_path: Path) -> None:
        """Pins the 3-layer floor's inclusive side: issue #1 exists in every
        project, so the floor must reject 0 without rejecting 1. Without this
        the floor is pinned from one direction only and tightening it to
        `>= 2` drops #1 with the suite green — the same one-sided gap the
        2-layer floor had before its own adjacent-value case was added."""
        issue_dir = tmp_path / "planning" / "epic" / "milestone-01" / "issues" / "1-first-issue"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {1: 1}
        assert mr_counts == {}

    @pytest.mark.parametrize(
        "slug, expected",
        [
            ("2026-10-bit-color", {2026: 1}),
            ("1234567-thing", {1234567: 1}),
        ],
    )
    def test_three_layer_shape_does_not_apply_the_two_layer_disambiguation_guards(
        self, tmp_path: Path, slug: str, expected: dict
    ) -> None:
        """These two pass BY DECISION, and this test exists so that decision
        fails a build rather than living only in a comment.

        The date guard and the digit cap are disambiguation guards: they answer
        "is this component an id or a date?", a question that only arises where
        issue-folder-ness must be inferred from the component alone. The
        literal 'issues/' segment is an explicit declaration, so there is
        nothing to disambiguate under it — and extending those two guards there
        would stop both of these real issue folders being attributed. (The zero
        floor is different: it is a validity check, iids being 1-based, so it
        does apply to both shapes — see the two zero-id tests above.)

        If this test fails, someone extended a disambiguation guard into a
        branch that needs none; the fix is to revert that, not to change these
        expectations."""
        issue_dir = tmp_path / "planning" / "epic" / "milestone-01" / "issues" / slug
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == expected
        assert mr_counts == {}

    @pytest.mark.parametrize("slug", ["2026", "70"])
    def test_bare_digit_component_without_a_slug_is_not_read_as_an_issue(
        self, tmp_path: Path, slug: str
    ) -> None:
        """`_FLAT_ISSUE_COMPONENT_RE` requires a '-<slug>' suffix after the
        digits; a bare numeric folder name carries no suffix at all and must
        not be read as a 2-layer issue folder."""
        issue_dir = tmp_path / "planning" / "goal" / slug
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_milestone_directory_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        """'milestone-01-pipeline-budgets' carries digits but is not an issue
        folder. Reading one as issue #1 would invent a ticket and log hours
        against it."""
        milestone_dir = tmp_path / "planning" / "goal" / "milestone-01-pipeline-budgets"
        milestone_dir.mkdir(parents=True)
        target = milestone_dir / "status.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_step_folder_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        """In planning/<goal>/<work-slug>/<step>/ the third component is a step
        of a multi-step effort, not an issue — only one component deep inside
        the goal is eligible."""
        step_dir = tmp_path / "planning" / "goal" / "rollout" / "2-second-step"
        step_dir.mkdir(parents=True)
        target = step_dir / "design.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_date_named_folder_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        """A folder named for a date would otherwise read as issue 2026."""
        dated_dir = tmp_path / "planning" / "goal" / "2026-10-12-incident"
        dated_dir.mkdir(parents=True)
        target = dated_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    @pytest.mark.parametrize(
        "slug",
        [
            "2026-10-sprint-notes",
            "2026-1-5-incident",
            "2026-10",
            "2026-10-12",
            "2026-03-14-incident",
            "2026-09-30-postmortem",
            "2026-12-01-retro",
            "1999-11-20-incident",
        ],
    )
    def test_month_or_unpadded_date_named_folder_is_not_read_as_an_issue(
        self, tmp_path: Path, slug: str
    ) -> None:
        """A year-month folder ('2026-10-sprint-notes'), the unpadded
        calendar spelling ('2026-1-5-incident'), and the bare slugless forms
        ('2026-10', '2026-10-12') all collide with issue 2026 the same way a
        full YYYY-MM-DD folder does, so the date guard must reject all of
        them, not only the zero-padded, slugged form. The bare forms in
        particular pin the regex's `$` alternative — dropping it leaves every
        slugged fixture passing through the `[^0-9]` branch instead.
        '2026-03-14-incident' and '2026-09-30-postmortem' pin the
        zero-padded single-digit month spelling specifically — every other
        fixture here uses month '10' or the unpadded '1', so dropping the
        `0?` from the month alternation would leave this exact ISO-dated
        folder shape unguarded. '2026-12-01-retro' and '1999-11-20-incident'
        pin the other two halves of the documented range: narrowing the month
        alternation below 12, or dropping the `19` from the year, leaves each
        of those folders reading as issue 2026 and 1999 respectively."""
        dated_dir = tmp_path / "planning" / "goal" / slug
        dated_dir.mkdir(parents=True)
        target = dated_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    @pytest.mark.parametrize(
        "slug, expected",
        [
            ("1234-2fa-support", {1234: 1}),
            ("4821-3-way-merge", {4821: 1}),
            ("2048-10-bit-color", {}),
            ("2026-13-inch-display", {2026: 1}),
        ],
    )
    def test_date_guard_does_not_over_reject_an_issue_id_with_a_numeric_slug_opening(
        self, tmp_path: Path, slug: str, expected: dict
    ) -> None:
        """A 4-digit issue id followed by a 1-2 digit slug token ('2fa',
        '3-way') must not read as a year-month date — an optional day group
        once made a bare second numeric component enough to reject it.

        '2048-10-bit-color' is the documented, irreducible exception: it is
        structurally identical to '2026-10-sprint-notes', which must stay
        rejected, so no rule keyed on the component's own text can tell the
        two apart.

        '2026-13-inch-display' pins the month alternation's upper bound:
        '13' is not a valid month, so a real issue id in 1900-2099 whose
        slug opens with a 13-99 token must still be attributed. Every other
        fixture here has a year outside 1900-2099, so this is the only case
        that depends on the upper bound alone.
        """
        issue_dir = tmp_path / "planning" / "goal" / slug
        issue_dir.mkdir(parents=True)
        target = issue_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == expected

    def test_tickets_directory_is_not_read_as_an_issue(self, tmp_path: Path) -> None:
        tickets_dir = tmp_path / "planning" / "tickets" / "70-draft"
        tickets_dir.mkdir(parents=True)
        target = tickets_dir / "tickets.yaml"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_file_under_the_legacy_top_level_reviews_directory_is_not_read_as_an_issue(
        self, tmp_path: Path
    ) -> None:
        """planning/reviews/... is the legacy top-level MR-review folder the
        old-format migration leaves on disk. A stray non-YAML file directly
        under it must not be claimed as an issue — even though its own
        slug component ('70-foo') would match the 2-layer pattern on its
        own, which is what makes this fixture able to fail."""
        reviews_dir = tmp_path / "planning" / "reviews" / "70-foo"
        reviews_dir.mkdir(parents=True)
        target = reviews_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_file_under_a_reviews_subdirectory_of_a_two_layer_issue_folder_is_counted(
        self, tmp_path: Path
    ) -> None:
        """A legacy reviews/ subdirectory left inside a 2-layer issue folder
        (the planning convention explicitly leaves these in place) must
        still count as evidence for that issue — the non-issue-directory
        guard excludes planning/reviews/ at the top level, not every
        directory named 'reviews' anywhere deeper in the tree."""
        reviews_dir = tmp_path / "planning" / "goal" / "70-thing" / "reviews"
        reviews_dir.mkdir(parents=True)
        target = reviews_dir / "notes.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {70: 1}

    def test_three_layer_shape_still_wins_and_is_not_double_counted(self, tmp_path: Path) -> None:
        """A 3-layer path also has a numeric component one deep inside the
        goal in some trees; the matchers must not both fire on one file."""
        issue_dir = (
            tmp_path / "planning" / "goal" / "99-epicish" / "milestone-01" / "issues" / "42-real"
        )
        issue_dir.mkdir(parents=True)
        target = issue_dir / "design.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {42: 1}

    def test_file_directly_in_the_goal_folder_yields_nothing(self, tmp_path: Path) -> None:
        """planning/<goal>/<file> has no issue-folder component at all —
        the depth guard must reject it before the component regex gets a
        chance to, so the fixture's own basename ('70-overview.md') is
        chosen to match the 2-layer pattern if the depth guard slipped."""
        goal_dir = tmp_path / "planning" / "goal"
        goal_dir.mkdir(parents=True)
        target = goal_dir / "70-overview.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_progress_md_at_the_planning_root_yields_nothing(self, tmp_path: Path) -> None:
        """planning/progress.md is the most frequently touched file in
        every tree following this convention, and the 2-component path
        (planning/<file>, with no goal folder at all) had no test of its
        own — it must not raise and must attribute nothing."""
        planning_dir = tmp_path / "planning"
        planning_dir.mkdir(parents=True)
        target = planning_dir / "progress.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_mr_review_yaml_is_detected(self, tmp_path: Path) -> None:
        reviews_dir = tmp_path / "planning" / "epic" / "reviews"
        reviews_dir.mkdir(parents=True)
        target = reviews_dir / "MR121-review.yaml"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {121: 1}

    def test_mr_review_yaml_inside_a_two_layer_issue_folder_is_not_also_counted_as_the_issue(
        self, tmp_path: Path
    ) -> None:
        """reviews/MR<N>-review.yaml is claimed by the MR matcher first and
        the branch must stop there — without the `continue` after it, this
        file would also fall through to the 2-layer matcher and be double
        counted as issue 70, since 'reviews' only guards the top-level
        planning/reviews/ shape, not a reviews/ subdirectory one level
        inside an issue folder."""
        reviews_dir = tmp_path / "planning" / "goal" / "70-epic" / "reviews"
        reviews_dir.mkdir(parents=True)
        target = reviews_dir / "MR5-review.yaml"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {5: 1}

    def test_multiple_files_in_the_same_issue_folder_sum_into_one_count(
        self, tmp_path: Path
    ) -> None:
        issue_dir = tmp_path / "planning" / "epic" / "issues" / "7-foo"
        issue_dir.mkdir(parents=True)
        for name in ("design.md", "code-review.md"):
            target = issue_dir / name
            target.write_text("x")
            _set_mtime(target, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, _mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {7: 2}

    def test_file_mtimed_on_a_different_date_is_excluded(self, tmp_path: Path) -> None:
        issue_dir = tmp_path / "planning" / "epic" / "issues" / "7-foo"
        issue_dir.mkdir(parents=True)
        target = issue_dir / "design.md"
        target.write_text("x")
        _set_mtime(target, _TARGET_DAY - timedelta(days=5))

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}

    def test_repository_with_no_planning_directory_yields_nothing(self, tmp_path: Path) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        issue_counts, mr_counts = handler._scan_planning_tree(_TARGET_DAY)  # noqa: SLF001

        assert issue_counts == {}
        assert mr_counts == {}


# ---------------------------------------------------------------------------
# Whole-report assembly
# ---------------------------------------------------------------------------


class TestReportAssembly:
    def test_clean_repo_reports_zero_activity_at_exit_zero_shape(self, tmp_path: Path) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=""),
        ):
            report = handler.report("2026-08-13")

        assert report == ActivityReport(_TARGET_DAY, "repo", [], 0, 0, None)

    def test_repo_name_is_the_toplevel_directory_name(self, tmp_path: Path) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(toplevel="/home/user/projects/alpha-svc\n", reflog="", status=""),
        ):
            report = handler.report("2026-08-13")

        assert report.repo == "alpha-svc"

    def test_default_date_is_todays_local_date(self, tmp_path: Path) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        handler._now = lambda: datetime(2026, 8, 13, 15, 0, 0, tzinfo=_FIXED_TZ)  # noqa: SLF001

        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=""),
        ):
            report = handler.report(None)

        assert report.activity_date == _TARGET_DAY

    def test_multiple_reflog_issues_are_sorted_by_id(self) -> None:
        reflog = "\n".join(
            [
                _reflog_line("a", "2026-08-13T09:00:00+06:00", "commit: x. Ref #20"),
                _reflog_line("b", "2026-08-13T10:00:00+06:00", "commit: y. Ref #5"),
            ]
        )
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/repo"))
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [
            IssueActivity(5, "reflog", 1),
            IssueActivity(20, "reflog", 1),
        ]

    def test_issue_corroborated_by_reflog_and_planning_reports_both(self, tmp_path: Path) -> None:
        """An issue attributed by two independent sources on the same date must
        surface both entries — not just the first one found, unlike the
        reflog/branch pair where branch is deliberately suppressed once the
        reflog already fired."""
        issue_dir = tmp_path / "planning" / "epic" / "issues" / "10-foo"
        issue_dir.mkdir(parents=True)
        design = issue_dir / "design.md"
        design.write_text("x")
        _set_mtime(design, _TARGET_DAY)

        reflog = _reflog_line("aaa1111", "2026-08-13T09:00:00+06:00", "commit: work. Ref #10")
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch("projctl.handlers.activity.subprocess.run", _git_stub(reflog=reflog)):
            report = handler.report("2026-08-13")

        assert report.issues == [
            IssueActivity(10, "reflog", 1),
            IssueActivity(10, "planning", 1),
        ]

    def test_planning_and_review_sources_fire_with_no_reflog_or_branch_evidence(
        self, tmp_path: Path
    ) -> None:
        """Sources 3 and 4 must run even on a day with no commit and no branch
        match — the whole point of adding them is to catch design/review work
        that source 1 and 2 are structurally blind to."""
        issue_dir = tmp_path / "planning" / "epic" / "issues" / "3-foo"
        issue_dir.mkdir(parents=True)
        design = issue_dir / "design.md"
        design.write_text("x")
        _set_mtime(design, _TARGET_DAY)

        reviews_dir = tmp_path / "planning" / "epic" / "reviews"
        reviews_dir.mkdir(parents=True)
        review = reviews_dir / "MR55-review.yaml"
        review.write_text("x")
        _set_mtime(review, _TARGET_DAY)

        handler = ActivityHandler(tz=_FIXED_TZ, cwd=tmp_path)
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(reflog="", status=""),
        ):
            report = handler.report("2026-08-13")

        assert report.issues == [
            IssueActivity(3, "planning", 1),
            IssueActivity(55, "review", 1),
        ]


# ---------------------------------------------------------------------------
# Errors: not a git repo, git missing, bad date
# ---------------------------------------------------------------------------


class TestGitTransportErrors:
    def test_not_a_git_repository_raises_platform_error(self) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/tmp"))
        with patch(
            "projctl.handlers.activity.subprocess.run",
            side_effect=lambda cmd, **kw: _completed(cmd, 128, "", "fatal: not a git repository"),
        ):
            with pytest.raises(PlatformError, match="not a git repository"):
                handler.report("2026-08-13")

    def test_empty_toplevel_path_raises_platform_error(self) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/tmp"))
        with patch(
            "projctl.handlers.activity.subprocess.run",
            _git_stub(toplevel=""),
        ):
            with pytest.raises(PlatformError, match="returned no path"):
                handler.report("2026-08-13")

    def test_git_not_installed_raises_platform_error(self) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/tmp"))
        with patch(
            "projctl.handlers.activity.subprocess.run",
            side_effect=FileNotFoundError("git"),
        ):
            with pytest.raises(PlatformError, match="git executable not found"):
                handler.report("2026-08-13")


class TestNow:
    def test_production_now_with_no_injected_tz_is_aware(self) -> None:
        """With tz=None (the production path), _now() must still return an aware
        datetime — see the docstring on the naive/aware hazard this avoids."""
        handler = ActivityHandler()
        assert handler._now().tzinfo is not None  # noqa: SLF001


class TestDateParsing:
    def test_invalid_date_raises_value_error(self) -> None:
        handler = ActivityHandler(tz=_FIXED_TZ, cwd=Path("/tmp"))
        with pytest.raises(ValueError, match="not-a-date"):
            handler.report("not-a-date")


# ---------------------------------------------------------------------------
# JSON / table rendering shape
# ---------------------------------------------------------------------------


class TestActivityReportToDict:
    def test_json_shape_matches_documented_contract(self) -> None:
        report = ActivityReport(
            _TARGET_DAY,
            "alpha-svc",
            [IssueActivity(4, "branch", 3)],
            unattributed_events=0,
            unattributed_files=0,
            unattributed_branch=None,
        )

        payload = report.to_dict()

        assert payload == {
            "date": "2026-08-13",
            "repo": "alpha-svc",
            "issues": [{"id": 4, "via": "branch", "count": 3}],
            "unattributed": {"events": 0, "files": 0, "branch": None},
        }

    def test_count_key_is_uniform_across_every_via(self) -> None:
        """The count key name used to depend on `via` ('events' for reflog,
        'files' for branch); now that there are four sources, it must be the
        single uniform key 'count' for all of them."""
        for via in ("reflog", "branch", "planning", "review"):
            report = ActivityReport(_TARGET_DAY, "repo", [IssueActivity(1, via, 3)], 0, 0, None)
            payload = report.to_dict()
            assert payload["issues"] == [{"id": 1, "via": via, "count": 3}]

    def test_json_is_serializable(self) -> None:
        report = ActivityReport(
            _TARGET_DAY,
            "repo",
            [IssueActivity(4, "branch", 1)],
            unattributed_events=2,
            unattributed_files=0,
            unattributed_branch="main",
        )

        json.dumps(report.to_dict())  # must not raise


class TestActivityReportPrintTable:
    def test_prints_issues_and_unattributed_sections(self, capsys: pytest.CaptureFixture) -> None:
        report = ActivityReport(
            _TARGET_DAY,
            "repo",
            [IssueActivity(4, "branch", 3)],
            unattributed_events=1,
            unattributed_files=2,
            unattributed_branch="main",
        )

        report.print_table()

        out = capsys.readouterr().out
        assert "#4" in out
        assert "3 file" in out
        assert "1 reflog event" in out
        assert "2 modified file" in out
        assert "'main'" in out

    def test_review_source_is_prefixed_with_bang_not_hash(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """via="review" holds a merge-request number, not an issue number — the
        table must not print it as '#55', which would read as an issue."""
        report = ActivityReport(_TARGET_DAY, "repo", [IssueActivity(55, "review", 1)], 0, 0, None)

        report.print_table()

        out = capsys.readouterr().out
        assert "!55" in out
        assert "#55" not in out

    def test_prints_no_attributed_issues_notice_when_empty(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        report = ActivityReport(_TARGET_DAY, "repo", [], 0, 0, None)

        report.print_table()

        assert "No attributed issues." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _args(**overrides) -> SimpleNamespace:
    defaults = {"date": None, "json": False}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestArgparseWiring:
    """Exercises the real parser built by _add_activity_subparser(), not a hand-rolled args object."""

    def test_real_parser_wires_bare_command_to_handler_report(self) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.return_value = ActivityReport(
                _TARGET_DAY, "repo", [], 0, 0, None
            )

            rc = main(["activity"])

        assert rc == 0
        mock_handler_cls.return_value.report.assert_called_once_with(None)

    def test_real_parser_wires_date_positional(self) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.return_value = ActivityReport(
                _TARGET_DAY, "repo", [], 0, 0, None
            )

            rc = main(["activity", "2026-08-13"])

        assert rc == 0
        mock_handler_cls.return_value.report.assert_called_once_with("2026-08-13")

    def test_real_parser_wires_json_flag(self, capsys: pytest.CaptureFixture) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.return_value = ActivityReport(
                _TARGET_DAY, "beta-svc", [IssueActivity(464, "reflog", 3)], 0, 0, None
            )

            rc = main(["activity", "2026-08-13", "--json"])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "date": "2026-08-13",
            "repo": "beta-svc",
            "issues": [{"id": 464, "via": "reflog", "count": 3}],
            "unattributed": {"events": 0, "files": 0, "branch": None},
        }


class TestCmdActivityOutputSelection:
    def test_default_output_is_the_human_table_not_json(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.return_value = ActivityReport(
                _TARGET_DAY, "repo", [IssueActivity(4, "branch", 1)], 0, 0, None
            )

            rc = cmd_activity(_args())

        assert rc == 0
        out = capsys.readouterr().out
        assert "#4" in out
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)

    def test_json_flag_emits_valid_json_and_no_table_text(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.return_value = ActivityReport(
                _TARGET_DAY, "repo", [IssueActivity(4, "branch", 1)], 0, 0, None
            )

            rc = cmd_activity(_args(json=True))

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["issues"] == [{"id": 4, "via": "branch", "count": 1}]


class TestErrorHandling:
    """Handler failures become a non-zero exit code, not a traceback."""

    def test_platform_error_returns_one(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.side_effect = PlatformError("not a git repository")

            with caplog.at_level("ERROR"):
                rc = cmd_activity(_args())

        assert rc == 1
        assert "not a git repository" in caplog.text

    def test_value_error_returns_one(self, caplog: pytest.LogCaptureFixture) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.side_effect = ValueError("Invalid date")

            with caplog.at_level("ERROR"):
                rc = cmd_activity(_args(date="not-a-date"))

        assert rc == 1
        assert "Invalid date" in caplog.text


class TestNoActivityIsSuccessNotError:
    def test_no_activity_found_still_exits_zero(self) -> None:
        with patch("projctl.cli.ActivityHandler") as mock_handler_cls:
            mock_handler_cls.return_value.report.return_value = ActivityReport(
                _TARGET_DAY, "repo", [], 0, 0, None
            )

            rc = cmd_activity(_args())

        assert rc == 0
