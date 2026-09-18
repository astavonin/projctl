"""Tests for the 'load mr --comments --json' CLI wiring (projctl.cli.cmd_load)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from projctl.cli import cmd_load, main


def _write_platform_config(path: Path, platform: str) -> Path:
    cfg_path = path / "config.yaml"
    data: Dict[str, Any] = {"platform": platform}
    if platform == "github":
        data["github"] = {"repo": "owner/test-repo"}
    else:
        data["gitlab"] = {"default_group": "test/group"}
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return cfg_path


def _args(**kwargs: Any) -> SimpleNamespace:
    defaults: Dict[str, Any] = {
        "config": None,
        "resource_type": "mr",
        "reference": "235",
        "comments": False,
        "json": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestCmdLoadJsonValidation:
    """--json is scoped to 'load mr --comments' on GitLab."""

    def test_json_without_comments_is_rejected(self, tmp_path: Path, capsys) -> None:
        cfg = _write_platform_config(tmp_path, "gitlab")
        args = _args(config=str(cfg), json=True, comments=False)

        result = cmd_load(args)

        assert result == 1
        assert "--comments" in capsys.readouterr().err

    def test_json_on_non_mr_resource_is_rejected(self, tmp_path: Path, capsys) -> None:
        cfg = _write_platform_config(tmp_path, "gitlab")
        args = _args(
            config=str(cfg), resource_type="issue", reference="42", json=True, comments=True
        )

        result = cmd_load(args)

        assert result == 1
        assert "load mr --comments" in capsys.readouterr().err

    def test_json_on_github_platform_is_rejected(self, tmp_path: Path, capsys) -> None:
        cfg = _write_platform_config(tmp_path, "github")
        args = _args(config=str(cfg), json=True, comments=True)

        result = cmd_load(args)

        assert result == 1
        assert "GitLab" in capsys.readouterr().err

    def test_plain_load_on_github_is_never_rejected_by_the_json_guard(self, tmp_path: Path) -> None:
        """A plain load (json=False) must go through untouched, whatever
        resource_type or platform it targets."""
        cfg = _write_platform_config(tmp_path, "github")
        args = _args(config=str(cfg), resource_type="issue", reference="42", json=False)

        with patch("projctl.cli.GithubLoader") as mock_loader_cls:
            mock_loader_cls.return_value = MagicMock()
            result = cmd_load(args)

        assert result == 0


class TestCmdLoadJsonDispatch:
    """A valid --json invocation prints load_mr_comments_json()'s payload as JSON."""

    def test_json_dispatch_prints_the_loader_payload(self, tmp_path: Path, capsys) -> None:
        cfg = _write_platform_config(tmp_path, "gitlab")
        args = _args(config=str(cfg), json=True, comments=True)
        payload = {"mr": {"viewer": "astavonin"}, "threads": []}

        with patch("projctl.cli.TicketLoader") as mock_loader_cls:
            instance = mock_loader_cls.return_value
            instance.load_mr_comments_json.return_value = payload
            result = cmd_load(args)

        assert result == 0
        instance.load_mr_comments_json.assert_called_once_with("235")
        assert json.loads(capsys.readouterr().out) == payload

    def test_plain_comments_load_still_uses_the_markdown_path(self, tmp_path: Path) -> None:
        """comments=True, json=False must still route through load_mr_comments()
        / print_mr_info() — --json is additive, not a new default."""
        cfg = _write_platform_config(tmp_path, "gitlab")
        args = _args(config=str(cfg), json=False, comments=True)

        with patch("projctl.cli.TicketLoader") as mock_loader_cls:
            instance = mock_loader_cls.return_value
            instance.load_mr_comments.return_value = {"mr": {}, "comments": []}
            result = cmd_load(args)

        assert result == 0
        instance.load_mr_comments.assert_called_once_with("235")
        instance.load_mr_comments_json.assert_not_called()


_MR_VIEW: Dict[str, Any] = {
    "iid": 235,
    "title": "Ref #42: fix the thing",
    "web_url": "https://gitlab.example.com/group/project/-/merge_requests/235",
    "author": {"username": "author.user", "name": "Author Display Name"},
    "source_project_id": 10,
    "target_project_id": 20,
    "source_branch": "feature/42-fix",
    "target_branch": "main",
}


def _current_user(username: str = "astavonin") -> str:
    return json.dumps({"data": {"currentUser": {"username": username}}})


def _note(note_id: int, body: str = "x") -> Dict[str, Any]:
    return {
        "id": note_id,
        "system": False,
        "body": body,
        "author": {"username": "reviewer", "name": "Reviewer"},
        "resolvable": True,
        "resolved": False,
        "created_at": "2026-08-07T09:00:00Z",
    }


def _run_cmd_load_json(tmp_path: Path, mock_run: Mock, responses: list) -> int:
    """Drive `cmd_load()` for a `load mr --comments --json` invocation against a
    mocked `subprocess.run`, patched with `responses` as the call-order replay.
    """
    cfg = _write_platform_config(tmp_path, "gitlab")
    args = _args(config=str(cfg), json=True, comments=True)
    mock_run.side_effect = responses
    return cmd_load(args)


class TestCmdLoadJsonFailureModes:
    """cmd_load() driven through subprocess.run against a real TicketLoader,
    rather than a mocked-out one — test_cli_load_json.py's other classes all
    patch TicketLoader wholesale, so none of them exercise cmd_load()
    against an actually-failing loader. tests/regression/test_cli_parity.py
    sets this convention for the issue path."""

    @patch("subprocess.run")
    def test_null_current_user_returns_rc_1(
        self, mock_run: Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `currentUser: null` GraphQL response is a PlatformError, already
        caught by cmd_load()'s existing except clause — rc 1, not a
        traceback."""
        with caplog.at_level("ERROR"):
            result = _run_cmd_load_json(
                tmp_path,
                mock_run,
                [
                    Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
                    Mock(stdout=json.dumps([]), returncode=0),
                    Mock(stdout=json.dumps({"data": {"currentUser": None}}), returncode=0),
                ],
            )

        assert result == 1
        assert "currentUser" in caplog.text
        # The json branch's `except KeyError` must not widen to catch this: the
        # wrapper interpolates {err}, so a substring check alone cannot see it.
        assert "malformed note payload" not in caplog.text

    @patch("subprocess.run")
    def test_note_missing_id_returns_rc_1_not_a_traceback(
        self, mock_run: Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A note payload missing its own `id` raises KeyError inside the
        fold; cmd_load must catch it and print a one-line error naming the
        malformed note payload, not let it escape main() as a traceback."""
        broken_note = _note(1)
        del broken_note["id"]
        discussion = [{"id": "d1", "notes": [broken_note]}]

        with caplog.at_level("ERROR"):
            result = _run_cmd_load_json(
                tmp_path,
                mock_run,
                [
                    Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
                    Mock(stdout=json.dumps(discussion), returncode=0),
                    Mock(stdout=_current_user(), returncode=0),
                ],
            )

        assert result == 1
        assert "malformed note payload" in caplog.text.lower()

    @patch("subprocess.run")
    def test_null_body_note_does_not_escape_as_a_traceback(
        self, mock_run: Mock, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A note payload with `"body": null` used to raise AttributeError
        out of `_iter_filtered_notes()`'s `.strip()` call, escaping every
        `cmd_load()` except arm as a traceback. Fixed, the note is treated
        like a blank-body one and filtered out of the fold; with no other
        note in the discussion, the thread never appears and cmd_load
        succeeds (rc 0)."""
        broken_note = _note(1)
        broken_note["body"] = None
        discussion = [{"id": "d1", "notes": [broken_note]}]

        result = _run_cmd_load_json(
            tmp_path,
            mock_run,
            [
                Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
                Mock(stdout=json.dumps(discussion), returncode=0),
                Mock(stdout=_current_user(), returncode=0),
            ],
        )

        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["threads"] == []

    @patch("subprocess.run")
    def test_null_notes_list_does_not_escape_as_a_traceback(
        self, mock_run: Mock, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A discussion with `"notes": null` (not an absent key or `[]`) must
        not raise `TypeError: 'NoneType' object is not iterable` uncaught by
        cmd_load()'s except clauses — it degrades to a discussion with no
        notes, the same way a missing `notes` key already does."""
        discussion = [{"id": "d1", "notes": None}]

        result = _run_cmd_load_json(
            tmp_path,
            mock_run,
            [
                Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
                Mock(stdout=json.dumps(discussion), returncode=0),
                Mock(stdout=_current_user(), returncode=0),
            ],
        )

        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["threads"] == []

    @patch("subprocess.run")
    def test_null_discussion_list_payload_does_not_escape_as_a_traceback(
        self, mock_run: Mock, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A `null` top-level note-list payload (glab emitting `null` in
        place of `[]`) must not raise `TypeError: 'NoneType' object is not
        iterable` uncaught by cmd_load()'s except clauses — it degrades to
        no discussions at all."""
        result = _run_cmd_load_json(
            tmp_path,
            mock_run,
            [
                Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
                Mock(stdout=json.dumps(None), returncode=0),
                Mock(stdout=_current_user(), returncode=0),
            ],
        )

        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["threads"] == []


class TestCmdLoadJsonSuccessPaths:
    """cmd_load() success-path behavior driven through a real TicketLoader,
    rather than a mocked-out one — see TestCmdLoadJsonFailureModes for why."""

    @patch("subprocess.run")
    def test_offset_less_created_at_is_skipped_as_unusable(
        self, mock_run: Mock, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """An offset-less created_at ("no Z, no +HH:MM") is unusable rather
        than a crash: the thread is skipped and cmd_load succeeds (rc 0).
        Reconstructing the pre-C3 code and driving cmd_load() with this
        exact single-note fixture measured rc 0 with skip_reason == "" —
        a one-element sorted() performs zero comparisons, so the TypeError
        a mixed-tz-awareness sort raises needs at least two notes (see
        test_fold_does_not_raise_when_naive_and_aware_created_at_are_mixed
        in test_loader.py for that case)."""
        note = _note(1)
        note["created_at"] = "2026-08-07T09:00:00"
        discussion = [{"id": "d1", "notes": [note]}]

        result = _run_cmd_load_json(
            tmp_path,
            mock_run,
            [
                Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
                Mock(stdout=json.dumps(discussion), returncode=0),
                Mock(stdout=_current_user(), returncode=0),
            ],
        )

        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["threads"][0]["skip_reason"] == "unusable_created_at"


class TestCmdLoadJsonThroughMain:
    """One test that drives argparse for real — every other --json test in
    this module calls cmd_load() directly with a hand-built SimpleNamespace,
    none of which exercise the parser, so a mutant deleting the --json
    argument registration would still leave every one of them green."""

    @patch("subprocess.run")
    def test_json_flag_through_main_parses_and_carries_folded_threads(
        self, mock_run: Mock, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = _write_platform_config(tmp_path, "gitlab")
        discussion = [{"id": "d1", "notes": [_note(1, "claim")]}]
        mock_run.side_effect = [
            Mock(stdout=json.dumps(_MR_VIEW), returncode=0),
            Mock(stdout=json.dumps(discussion), returncode=0),
            Mock(stdout=_current_user(), returncode=0),
        ]

        rc = main(["--config", str(cfg), "load", "mr", "235", "--comments", "--json"])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["mr"]["viewer"] == "astavonin"
        assert len(payload["threads"]) == 1
        assert payload["threads"][0]["discussion_id"] == "d1"


class TestCmdLoadKeyErrorScope:
    """The `except KeyError` in `_dispatch_load()`'s mr --json branch names the
    malformed-note case specifically — it must not be re-broadened onto
    `cmd_load()`'s outer try, which would catch a KeyError raised by any
    load path and misreport it as a malformed MR note payload."""

    def test_key_error_on_a_non_mr_load_path_is_not_reported_as_a_malformed_note_payload(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = _write_platform_config(tmp_path, "gitlab")
        args = _args(config=str(cfg), resource_type="issue", reference="42", json=False)

        with patch("projctl.cli.TicketLoader") as mock_loader_cls:
            instance = mock_loader_cls.return_value
            instance.load_ticket_with_epic.side_effect = KeyError("iid")

            with caplog.at_level("ERROR"):
                with pytest.raises(KeyError):
                    cmd_load(args)

        assert "malformed note payload" not in caplog.text.lower()
