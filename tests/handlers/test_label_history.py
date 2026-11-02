"""Tests for projctl.handlers.label_history module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from projctl.cli import main as cli_main
from projctl.exceptions import PlatformError
from projctl.handlers.label_history import LabelEvent, LabelHistoryHandler

# Patch the shared runner rather than the handler: label_history.py reads through
# glab_runner.run_glab_json_pages, so this one point covers execute and parse.
_PATCH_PATH = "projctl.utils.glab_runner.run_glab_command"


def _make_handler() -> LabelHistoryHandler:
    return LabelHistoryHandler(config=MagicMock())


def _event(date: str, action: str, name: str, user: str = "someone") -> dict:
    """Build one API event object in the shape the endpoint returns."""
    return {
        "id": 1,
        "created_at": f"{date}T05:37:20.404Z",
        "action": action,
        "label": {"id": 7, "name": name},
        "user": {"username": user},
    }


def _endpoint_from_call(mock_run: MagicMock) -> str:
    return mock_run.call_args[0][0][-1]


class TestIssueEvents:
    """Tests for LabelHistoryHandler.issue_events."""

    @patch(_PATCH_PATH)
    def test_events_are_parsed_into_records(self, mock_run: MagicMock) -> None:
        mock_run.return_value = json.dumps(
            [_event("2026-09-28", "add", "Iteration::2", "alex.stavonin")]
        )

        events = _make_handler().issue_events("565")

        assert events == [
            LabelEvent(
                date="2026-09-28",
                action="add",
                label="Iteration::2",
                username="alex.stavonin",
            )
        ]

    @patch(_PATCH_PATH)
    def test_local_ref_uses_the_fullpath_placeholder(self, mock_run: MagicMock) -> None:
        mock_run.return_value = "[]"

        _make_handler().issue_events("565")

        assert _endpoint_from_call(mock_run) == (
            "projects/:fullpath/issues/565/resource_label_events"
        )

    @patch(_PATCH_PATH)
    def test_url_ref_targets_its_own_project(self, mock_run: MagicMock) -> None:
        mock_run.return_value = "[]"

        _make_handler().issue_events("https://host/group/sub/proj/-/issues/42")

        endpoint = _endpoint_from_call(mock_run)
        assert "group%2Fsub%2Fproj" in endpoint
        assert endpoint.endswith("/issues/42/resource_label_events")

    @patch(_PATCH_PATH)
    def test_label_prefix_filters_other_scopes_out(self, mock_run: MagicMock) -> None:
        mock_run.return_value = json.dumps(
            [
                _event("2026-09-28", "add", "Iteration::2"),
                _event("2026-09-28", "add", "type::feature"),
            ]
        )

        events = _make_handler().issue_events("565", label_prefix="Iteration::")

        assert [e.label for e in events] == ["Iteration::2"]

    @patch(_PATCH_PATH)
    def test_event_for_a_deleted_label_is_skipped_not_blank(self, mock_run: MagicMock) -> None:
        mock_run.return_value = json.dumps(
            [
                {"id": 1, "created_at": "2026-09-28T00:00:00Z", "action": "add", "label": None},
                _event("2026-09-29", "add", "Iteration::2"),
            ]
        )

        events = _make_handler().issue_events("565")

        assert [e.label for e in events] == ["Iteration::2"]

    @patch(_PATCH_PATH)
    def test_non_object_item_is_an_error_not_a_silent_drop(self, mock_run: MagicMock) -> None:
        mock_run.return_value = json.dumps(["not-an-object"])

        with pytest.raises(PlatformError, match="Expected label event objects"):
            _make_handler().issue_events("565")

    @pytest.mark.parametrize(
        "item",
        [
            {"id": 1, "created_at": "2026-01-01T00:00:00Z", "action": "add", "label": "Scope::2"},
            {
                "id": 2,
                "created_at": "2026-01-01T00:00:00Z",
                "action": "add",
                "label": {"name": "Scope::2"},
                "user": "someone",
            },
        ],
    )
    @patch(_PATCH_PATH)
    def test_a_non_object_label_or_user_is_an_error_not_a_crash(
        self, mock_run: MagicMock, item: dict
    ) -> None:
        """`label: null` is a real shape handled as a deleted label, but a
        non-null non-object would reach .get() and raise AttributeError, which
        escapes the CLI's PlatformError/ValueError handler as a traceback."""
        mock_run.return_value = json.dumps([item])

        with pytest.raises(PlatformError, match="Expected label event objects"):
            _make_handler().issue_events("565")

    def test_unparseable_reference_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse issue reference"):
            _make_handler().issue_events("not-a-ref")


class TestFindRemoval:
    """Tests for LabelHistoryHandler.find_removal."""

    def test_a_label_removed_and_left_off_reports_the_removal(self) -> None:
        events = [
            _le("2026-09-28", "add", "Iteration::2"),
            _le("2026-10-23", "remove", "Iteration::2"),
        ]

        removal = LabelHistoryHandler.find_removal(events, "Iteration::2")

        assert removal is not None
        assert removal.date == "2026-10-23"

    def test_a_label_removed_then_re_added_has_not_left(self) -> None:
        """A re-add cancels an earlier removal.

        Six issues in one real project carried this exact shape — a scope
        label dropped and restored — and reporting the stale removal listed
        them as deferred while they were still in scope.
        """
        events = [
            _le("2026-07-16", "add", "Iteration::2"),
            _le("2026-09-11", "remove", "Iteration::2"),
            _le("2026-10-05", "add", "Iteration::2"),
        ]

        assert LabelHistoryHandler.find_removal(events, "Iteration::2") is None

    def test_the_last_of_several_removals_is_the_one_reported(self) -> None:
        events = [
            _le("2026-07-16", "add", "Iteration::2"),
            _le("2026-09-11", "remove", "Iteration::2"),
            _le("2026-10-05", "add", "Iteration::2"),
            _le("2026-10-28", "remove", "Iteration::2"),
        ]

        removal = LabelHistoryHandler.find_removal(events, "Iteration::2")

        assert removal is not None
        assert removal.date == "2026-10-28"

    def test_a_label_never_applied_has_no_removal(self) -> None:
        events = [_le("2026-09-28", "add", "Iteration::3")]

        assert LabelHistoryHandler.find_removal(events, "Iteration::2") is None

    def test_another_label_s_removal_does_not_count(self) -> None:
        events = [
            _le("2026-09-28", "add", "Iteration::2"),
            _le("2026-10-23", "remove", "Iteration::3"),
        ]

        assert LabelHistoryHandler.find_removal(events, "Iteration::2") is None


class TestEpicRejection:
    """Tests for the epic path, which GitLab does not serve."""

    def test_message_names_the_cause_and_the_alternative(self) -> None:
        message = LabelHistoryHandler.epic_unsupported_message("107")

        assert "&107" in message
        assert "404" in message
        assert "issues individually" in message


def _le(date: str, action: str, label: str, user: str = "someone") -> LabelEvent:
    """Build a LabelEvent directly, for the pure-function tests."""
    return LabelEvent(date=date, action=action, label=label, username=user)


class TestCmdHistory:
    """Tests for the `history` CLI command.

    The exit codes are a documented three-way contract — 0 success, 1 error,
    2 "--left found no removal in effect" — and a caller scanning a set of
    issues branches on them, so each one needs its own test. Conflating 1 and
    2 would turn an API failure into "it never left".
    """

    _EVENTS = [
        _event("2026-07-16", "add", "Scope::2"),
        _event("2026-10-23", "remove", "Scope::2"),
    ]

    @staticmethod
    def _run(argv: list, payload: str = "[]") -> int:
        with patch(_PATCH_PATH, return_value=payload):
            return cli_main(argv)

    def test_left_reports_the_removal_and_exits_zero(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = self._run(
            ["--config", str(new_config_path), "history", "issue", "565", "--left", "Scope::2"],
            json.dumps(self._EVENTS),
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "2026-10-23" in out
        assert "removed Scope::2" in out

    def test_a_label_still_applied_exits_two_saying_so(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Exit 2 with "still carries" — distinct from "never carried" below,
        because collapsing the two hides a re-add."""
        events = self._EVENTS + [_event("2026-10-28", "add", "Scope::2")]

        code = self._run(
            ["--config", str(new_config_path), "history", "issue", "565", "--left", "Scope::2"],
            json.dumps(events),
        )

        assert code == 2
        assert "still carries Scope::2" in capsys.readouterr().out

    def test_a_label_never_applied_exits_two_saying_so(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = self._run(
            ["--config", str(new_config_path), "history", "issue", "565", "--left", "Scope::9"],
            json.dumps(self._EVENTS),
        )

        assert code == 2
        assert "never carried Scope::9" in capsys.readouterr().out

    def test_left_json_form_carries_the_removal(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = self._run(
            [
                "--config",
                str(new_config_path),
                "history",
                "issue",
                "565",
                "--left",
                "Scope::2",
                "--json",
            ],
            json.dumps(self._EVENTS),
        )

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["removed"]["date"] == "2026-10-23"
        assert payload["removed"]["action"] == "remove"

    def test_timeline_renders_a_sign_per_action(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = self._run(
            ["--config", str(new_config_path), "history", "issue", "565"],
            json.dumps(self._EVENTS),
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "+Scope::2" in out
        assert "-Scope::2" in out

    def test_an_empty_timeline_says_so_rather_than_printing_nothing(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = self._run(["--config", str(new_config_path), "history", "issue", "565"])

        assert code == 0
        assert "No matching label events" in capsys.readouterr().out

    def test_an_epic_is_refused_with_the_reason(
        self, new_config_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exit 1, not an empty exit 0: GitLab serves no resource_label_events
        endpoint for group epics, and an empty timeline would read as "this
        epic never changed labels"."""
        with caplog.at_level("ERROR"):
            code = self._run(["--config", str(new_config_path), "history", "epic", "&107"])

        assert code == 1
        assert "&107" in caplog.text
        assert "issues individually" in caplog.text

    def test_a_bad_reference_exits_one_not_two(
        self, new_config_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reference that cannot be parsed is an error, not "no removal
        found" — a caller branching on 2 must not read it as a departure."""
        with caplog.at_level("ERROR"):
            code = self._run(
                [
                    "--config",
                    str(new_config_path),
                    "history",
                    "issue",
                    "not-a-ref",
                    "--left",
                    "Scope::2",
                ]
            )

        assert code == 1
        assert "Cannot parse issue reference" in caplog.text

    def test_an_api_failure_exits_one_not_two(
        self, new_config_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same reasoning as the bad reference: a failed fetch must never be
        reported as "the label is still applied"."""
        with caplog.at_level("ERROR"):
            with patch(_PATCH_PATH, side_effect=PlatformError("glab exploded")):
                code = cli_main(
                    [
                        "--config",
                        str(new_config_path),
                        "history",
                        "issue",
                        "565",
                        "--left",
                        "Scope::2",
                    ]
                )

        assert code == 1
        assert "glab exploded" in caplog.text

    def test_github_platform_is_refused(
        self, temp_dir: Path, new_config_data: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        github_cfg = dict(new_config_data)
        github_cfg["platform"] = "github"
        cfg_path = temp_dir / "github.yaml"
        with open(cfg_path, "w", encoding="utf-8") as handle:
            yaml.dump(github_cfg, handle)

        with caplog.at_level("ERROR"):
            code = self._run(["--config", str(cfg_path), "history", "issue", "565"])

        assert code == 1
        assert "only supported for GitLab" in caplog.text

    def test_the_fetch_is_paginated(self, new_config_path: Path) -> None:
        """An issue with more than one page of events must not silently report
        fewer, so --paginate has to reach glab."""
        with patch(_PATCH_PATH, return_value="[]") as mock_run:
            cli_main(["--config", str(new_config_path), "history", "issue", "565"])

        assert "--paginate" in mock_run.call_args[0][0]
