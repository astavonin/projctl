"""Tests for the CLI-level 'pipeline-debug' command (projctl.cli.cmd_pipeline_debug)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from projctl.cli import cmd_pipeline_debug, main


def _args(**overrides) -> SimpleNamespace:
    """Build an args namespace with every field cmd_pipeline_debug reads."""
    defaults = {"branch": "feature/x", "job_id": None, "job_name": None, "config": None}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(name="handler_cls")
def handler_cls_fixture():
    """Patch Config and PipelineHandler in the cli namespace; yield the class mock."""
    with patch("projctl.cli.Config", MagicMock()), patch("projctl.cli.PipelineHandler") as mock_cls:
        mock_cls.return_value.get_current_pipeline.return_value = {
            "id": 2030133,
            "status": "success",
            "web_url": "https://example.invalid/pipelines/2030133",
        }
        yield mock_cls


class TestJobNameSelection:
    """--job-name reaches a job by name at any status."""

    def test_passing_job_is_reachable_and_its_log_is_printed(
        self, handler_cls: Mock, capsys: pytest.CaptureFixture
    ) -> None:
        """A green job's log is printed.

        get_failed_jobs() filters to status == "failed", so without this path a job
        carrying allow_failure: true -- whose pipeline is green either way -- has no
        reachable log, and its own log is the only place its verdict exists.
        """
        handler = handler_cls.return_value
        handler.get_jobs_by_name.return_value = [
            {"id": 555, "name": "ota-e2e", "stage": "e2e", "status": "success", "duration": 592.1}
        ]
        handler.get_job_logs.return_value = "11 scenarios PASS"

        rc = cmd_pipeline_debug(_args(job_name="ota-e2e"))

        assert rc == 0
        handler.get_jobs_by_name.assert_called_once_with(2030133, "ota-e2e")
        assert "11 scenarios PASS" in capsys.readouterr().out

    def test_job_id_is_printed_so_it_can_be_passed_to_artifacts(
        self, handler_cls: Mock, capsys: pytest.CaptureFixture
    ) -> None:
        """The job ID appears in the header.

        `artifacts` requires --job-id and does not discover it, so this output is the
        documented way to get from a branch to that ID. The documentation states it,
        which makes printing it a contract rather than incidental formatting.
        """
        handler = handler_cls.return_value
        handler.get_jobs_by_name.return_value = [
            {"id": 555, "name": "ota-e2e", "stage": "e2e", "status": "success", "duration": 1.0}
        ]
        handler.get_job_logs.return_value = ""

        cmd_pipeline_debug(_args(job_name="ota-e2e"))

        assert "555" in capsys.readouterr().out

    def test_every_job_sharing_the_name_is_printed(
        self, handler_cls: Mock, capsys: pytest.CaptureFixture
    ) -> None:
        """A retry or matrix leg must not be silently dropped."""
        handler = handler_cls.return_value
        handler.get_jobs_by_name.return_value = [
            {"id": 1, "name": "ota-e2e", "stage": "e2e", "status": "failed", "duration": 1.0},
            {"id": 2, "name": "ota-e2e", "stage": "e2e", "status": "success", "duration": 2.0},
        ]
        handler.get_job_logs.side_effect = ["first attempt", "retry attempt"]

        cmd_pipeline_debug(_args(job_name="ota-e2e"))

        out = capsys.readouterr().out
        assert "first attempt" in out
        assert "retry attempt" in out

    def test_unmatched_name_is_reported_distinctly_and_exits_zero(
        self, handler_cls: Mock, capsys: pytest.CaptureFixture
    ) -> None:
        """A job gated out by rules: never ran -- not the same as running clean.

        Exit 0 because not running is a legitimate pipeline outcome, but the message
        must not read as an empty-but-successful log.
        """
        handler = handler_cls.return_value
        handler.get_jobs_by_name.return_value = []

        rc = cmd_pipeline_debug(_args(job_name="no-such-job"))

        assert rc == 0
        out = capsys.readouterr().out
        assert "No job named" in out
        assert "no-such-job" in out

    def test_failed_job_path_is_untouched_when_job_name_is_absent(
        self, handler_cls: Mock, capsys: pytest.CaptureFixture
    ) -> None:
        """Without --job-name the command still reports failed jobs only."""
        handler = handler_cls.return_value
        handler.get_failed_jobs.return_value = [
            {"id": 9, "name": "lint", "stage": "test", "status": "failed", "duration": 3.0}
        ]
        handler.get_job_logs.return_value = "lint exploded"

        rc = cmd_pipeline_debug(_args())

        assert rc == 0
        handler.get_jobs_by_name.assert_not_called()
        assert "lint exploded" in capsys.readouterr().out


class TestJobSelectorExclusivity:
    """--job-id and --job-name name different jobs and cannot both be honoured."""

    def test_job_id_with_job_name_is_rejected_at_parse_time(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Both selectors together exit 2 rather than silently dropping --job-name.

        --job-id skips pipeline discovery, so accepting both would print the job the
        user did not name while reporting nothing about the one they did.
        """
        with pytest.raises(SystemExit) as exc:
            main(["pipeline-debug", "--job-id", "5946580", "--job-name", "ota-e2e"])

        assert exc.value.code == 2
        assert "not allowed with" in capsys.readouterr().err
