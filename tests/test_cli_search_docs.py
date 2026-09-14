"""Tests for the 'search docs' CLI surface (projctl.cli) and the three
existing 'search' types' compatibility guarantees under the default-value
move that made docs possible (see design.md §5.1, §6 Compatibility table).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from projctl.cli import cmd_search, main
from projctl.handlers.github_search import GithubSearchHandler


def _write(path: Path, content: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Flag validation — docs rejects --state/--limit/--label, requires a query
# ---------------------------------------------------------------------------


class TestDocsFlagValidation:
    def test_label_is_rejected_under_docs(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["search", "docs", "q", "--label", "x"])
        assert rc == 1
        assert "--label" in capsys.readouterr().err

    def test_state_is_rejected_under_docs(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["search", "docs", "q", "--state", "opened"])
        assert rc == 1
        assert "--state" in capsys.readouterr().err

    def test_limit_is_rejected_under_docs(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["search", "docs", "q", "--limit", "5"])
        assert rc == 1
        assert "--limit" in capsys.readouterr().err

    def test_empty_query_is_rejected(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["search", "docs", ""])
        assert rc == 1
        assert "query" in capsys.readouterr().err

    def test_whitespace_only_query_is_rejected(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["search", "docs", "   "])
        assert rc == 1
        assert "query" in capsys.readouterr().err

    def test_bare_query_passes_validation_with_all_four_unset(self) -> None:
        with patch("projctl.cli.DocsSearchHandler") as mock_handler_cls:
            rc = cmd_search(_docs_args(query="cross toolchain sysroot"))
        assert rc == 0
        mock_handler_cls.return_value.search.assert_called_once_with(
            query="cross toolchain sysroot", related=False, config_path=None
        )

    def test_an_explicit_config_path_reaches_the_handler(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        with patch("projctl.cli.DocsSearchHandler") as mock_handler_cls:
            rc = cmd_search(_docs_args(query="q", config=str(cfg)))
        assert rc == 0
        mock_handler_cls.return_value.search.assert_called_once_with(
            query="q", related=False, config_path=cfg
        )


class TestRelatedRejectedForExistingTypes:
    @pytest.mark.parametrize("search_type", ["issues", "epics", "milestones"])
    def test_related_is_rejected(self, search_type: str, capsys: pytest.CaptureFixture) -> None:
        rc = main(["search", search_type, "q", "--related"])
        assert rc == 1
        assert "--related" in capsys.readouterr().err


def _docs_args(query: str = "q", related: bool = False, config: str | None = None):
    from types import SimpleNamespace

    return SimpleNamespace(
        type="docs", query=query, state=None, limit=None, label=None, related=related, config=config
    )


# ---------------------------------------------------------------------------
# Compatibility — the three existing types keep their dispatch and defaults
# ---------------------------------------------------------------------------


class TestSearchCompatibility:
    @patch("subprocess.run")
    def test_gitlab_issues_omitted_flags_still_read_all_and_20(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        mock_run.return_value = Mock(stdout="[]", returncode=0)
        rc = main(["--config", str(new_config_path), "search", "issues", "q"])
        assert rc == 0
        called_url = mock_run.call_args[0][0][-1]
        assert "per_page=20" in called_url
        assert "state=" not in called_url  # state=all omits the param entirely

    @patch("subprocess.run")
    def test_gitlab_issues_explicit_flags_reach_the_handler_unchanged(
        self, mock_run: Mock, new_config_path: Path
    ) -> None:
        mock_run.return_value = Mock(stdout="[]", returncode=0)
        rc = main(
            [
                "--config",
                str(new_config_path),
                "search",
                "issues",
                "q",
                "--state",
                "opened",
                "--limit",
                "5",
                "--label",
                "a",
                "--label",
                "b",
            ]
        )
        assert rc == 0
        called_url = mock_run.call_args[0][0][-1]
        assert "state=opened" in called_url
        assert "per_page=5" in called_url
        # The whole labels parameter, not a stray character the query already
        # puts in the URL — both labels, in order, comma-joined.
        assert "labels=a,b" in called_url

    @patch("subprocess.run")
    def test_github_issues_omitted_state_and_limit_still_apply_github_defaults(
        self, mock_run: Mock, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "projctl.yaml"
        cfg.write_text("platform: github\ngithub:\n  repo: org/repo\n")
        mock_run.return_value = Mock(stdout="[]", returncode=0)
        with patch.object(GithubSearchHandler, "search_issues") as mock_search:
            rc = main(["--config", str(cfg), "search", "issues", "q"])
        assert rc == 0
        mock_search.assert_called_once_with(query="q", state="all")

    def test_milestones_label_still_errors_unsupported(
        self, new_config_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rc = main(["--config", str(new_config_path), "search", "milestones", "q", "--label", "x"])
        assert rc == 1
        assert "not supported for milestones" in capsys.readouterr().err

    def test_github_epics_still_errors_unsupported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = tmp_path / "projctl.yaml"
        cfg.write_text("platform: github\ngithub:\n  repo: org/repo\n")
        with caplog.at_level("ERROR"):
            rc = main(["--config", str(cfg), "search", "epics", "q"])
        assert rc == 1
        assert "not supported on GitHub" in caplog.text

    def test_github_issues_state_active_still_errors_unsupported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = tmp_path / "projctl.yaml"
        cfg.write_text("platform: github\ngithub:\n  repo: org/repo\n")
        rc = main(["--config", str(cfg), "search", "issues", "q", "--state", "active"])
        assert rc == 1
        assert "milestone-only state" in capsys.readouterr().err

    def test_bogus_search_type_still_rejected_by_argparse_choices(self) -> None:
        with pytest.raises(SystemExit):
            main(["search", "bogus", "q"])

    def test_help_lists_docs_alongside_the_three_existing_types(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit):
            main(["search", "--help"])
        out = capsys.readouterr().out
        # The rendered choices list, not the prose: the prose names docs even
        # when the choice itself has been removed.
        assert "{issues,epics,milestones,docs}" in out


# ---------------------------------------------------------------------------
# CLI wiring for 'search docs'
# ---------------------------------------------------------------------------


class TestDocsSearchCliWiring:
    def test_no_config_file_runs_against_current_repo_with_defaults(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "sysroot"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "## Roadmap" in out

    def test_config_file_in_cwd_subdirectory_of_repo_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "projctl.yaml").write_text("search:\n  docs_path: nonexistent-configured\n")
        monkeypatch.chdir(subdir)

        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "sysroot"])
        assert rc == 0
        out = capsys.readouterr().out
        # The subdirectory config's configured docs_path would have produced a
        # "configured path is missing" warning if it had been read.
        assert "configured path is missing" not in out

    def test_nonexistent_explicit_config_is_a_hard_error(self, tmp_path: Path) -> None:
        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["--config", str(tmp_path / "nope.yaml"), "search", "docs", "sysroot"])
        assert rc == 1

    def test_malformed_search_section_in_callers_own_config_exits_one_no_partial_digest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        (tmp_path / "projctl.yaml").write_text("search: 3\n")
        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "sysroot"])
        assert rc == 1
        assert "## Roadmap" not in capsys.readouterr().out

    def test_a_null_search_section_in_the_callers_own_config_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        (tmp_path / "projctl.yaml").write_text("search:\n")
        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "sysroot"])
        assert rc == 1
        assert "## Roadmap" not in capsys.readouterr().out

    def test_query_matching_nothing_exits_zero_with_empty_prior_decisions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "zzzznomatch"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "## Roadmap" in out
        assert "## Prior decisions" in out

    def test_a_yaml_syntax_error_in_the_callers_own_config_exits_one_naming_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        (tmp_path / "projctl.yaml").write_text("not: valid: yaml: [\n")

        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            with caplog.at_level("ERROR"):
                rc = main(["search", "docs", "sysroot"])

        assert rc == 1
        assert "projctl.yaml" in caplog.text
        assert "## Roadmap" not in capsys.readouterr().out

    def test_an_unreadable_config_in_the_callers_own_repository_exits_one_naming_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        config = tmp_path / "projctl.yaml"
        config.write_text("search:\n  docs_path: docs\n")
        config.chmod(0o000)
        if os.access(config, os.R_OK):
            pytest.skip("the test process can read a 0o000 file (running as root)")

        try:
            with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
                with caplog.at_level("ERROR"):
                    rc = main(["search", "docs", "sysroot"])
        finally:
            config.chmod(0o644)

        assert rc == 1
        assert "projctl.yaml" in caplog.text
        assert "## Roadmap" not in capsys.readouterr().out

    def test_an_unreadable_explicit_config_exits_one_naming_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        config = tmp_path / "custom.yaml"
        config.write_text("search:\n  docs_path: docs\n")
        config.chmod(0o000)
        if os.access(config, os.R_OK):
            pytest.skip("the test process can read a 0o000 file (running as root)")

        try:
            with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
                with caplog.at_level("ERROR"):
                    rc = main(["--config", str(config), "search", "docs", "sysroot"])
        finally:
            config.chmod(0o644)

        assert rc == 1
        assert "custom.yaml" in caplog.text
        assert "## Roadmap" not in capsys.readouterr().out

    def test_an_explicit_config_path_naming_a_directory_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        directory = tmp_path / "custom.yaml"
        directory.mkdir()

        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            with caplog.at_level("ERROR"):
                rc = main(["--config", str(directory), "search", "docs", "sysroot"])

        assert rc == 1
        assert "custom.yaml" in caplog.text
        assert "## Roadmap" not in capsys.readouterr().out

    def test_a_directory_named_like_a_project_config_is_not_read_as_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # A probe answers whether there is a config file to read; a directory
        # holding the name is not one, and handing it to Config raised
        # IsADirectoryError from the path whose contract is "no config here".
        _write(tmp_path / "planning" / "notes.md", "sysroot text\n")
        (tmp_path / "projctl.yaml").mkdir()

        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "sysroot"])

        assert rc == 0
        assert "## Roadmap" in capsys.readouterr().out

    def test_a_repository_with_no_planning_directory_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        with patch("projctl.handlers.docs_search.get_repo_root", return_value=tmp_path):
            rc = main(["search", "docs", "sysroot"])
        assert rc == 0
        assert "## Prior decisions" in capsys.readouterr().out

    def test_a_config_outside_the_repository_root_supplies_the_search_section(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        repo = tmp_path / "repo"
        related = tmp_path / "related"
        _write(repo / "planning" / "a.md", "sysroot in the caller\n")
        _write(related / "planning" / "b.md", "sysroot in the related project\n")
        cfg = tmp_path / "config-dir" / "custom.yaml"
        # The relative entry resolves against the repository root, not against
        # this file's own directory.
        _write(cfg, "search:\n  related:\n    - ../related\n")

        with patch("projctl.handlers.docs_search.get_repo_root", return_value=repo):
            rc = main(["--config", str(cfg), "search", "docs", "sysroot", "--related"])

        assert rc == 0
        assert "related (undeclared)" in capsys.readouterr().out

    def test_one_absent_related_entry_does_not_abort_the_remaining_hops(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        present = tmp_path / "present"
        _write(caller / "planning" / "a.md", "sysroot in the caller\n")
        _write(present / "planning" / "b.md", "sysroot in the present project\n")
        (caller / "projctl.yaml").write_text(
            f"search:\n  related:\n    - ../does-not-exist\n    - {present}\n"
        )

        with patch("projctl.handlers.docs_search.get_repo_root", return_value=caller):
            rc = main(["search", "docs", "sysroot", "--related"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "does-not-exist: declared path is absent on this machine" in out
        assert "present (undeclared): planning" in out


# ---------------------------------------------------------------------------
# E2E — real git repositories under tmp_path
# ---------------------------------------------------------------------------


def _git_init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)


@pytest.mark.integration
class TestDocsSearchE2E:
    def test_related_run_against_two_real_repositories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        related = tmp_path / "related"
        _git_init(caller)
        _git_init(related)

        _write(caller / "planning" / "goal" / "overview.md", "## About\nThe caller's roadmap.\n")
        _write(caller / "planning" / "a.md", "cross toolchain sysroot in caller\n")
        (caller / "projctl.yaml").write_text(
            f"search:\n  related:\n    - {related}\n    - {tmp_path / 'absent'}\n"
        )

        _write(related / "planning" / "b.md", "cross toolchain sysroot in related\n")
        (related / "projctl.yaml").write_text("platform: github\n")

        monkeypatch.chdir(caller)
        rc = main(["search", "docs", "cross toolchain sysroot", "--related"])

        assert rc == 0
        out = capsys.readouterr().out
        roadmap, _, rest = out.partition("## Prior decisions")
        prior_decisions, _, footer = rest.partition("\n---\n")

        # One hit from each repository, each carrying its own platform tag —
        # asserted against the ranked half alone, since the footer names both
        # repositories whether or not a single section was indexed.
        assert "[untyped] planning/a.md" in prior_decisions
        assert "[untyped] planning/b.md" in prior_decisions
        assert "caller (undeclared)" in prior_decisions
        assert "related (github)" in prior_decisions

        assert "- planning/goal: The caller's roadmap." in roadmap
        assert len(roadmap.split()) <= 300

        assert "caller (undeclared): planning — 2 indexed files" in footer
        # The singular branch needs the line terminator: "1 indexed file" is a
        # substring of the plural form it is meant to be distinguished from.
        assert "related (github): planning — 1 indexed file\n" in footer
        assert "absent: declared path is absent on this machine" in footer

    def test_a_skipped_related_path_warns_on_stderr_while_stdout_stays_the_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        caller = tmp_path / "caller"
        _git_init(caller)
        _write(caller / "planning" / "a.md", "sysroot text\n")
        (caller / "projctl.yaml").write_text("search:\n  related:\n    - ../does-not-exist\n")
        monkeypatch.chdir(caller)

        root = logging.getLogger()
        saved = root.handlers[:]
        root.handlers.clear()
        try:
            rc = main(["search", "docs", "sysroot", "--related"])
        finally:
            root.handlers[:] = saved

        assert rc == 0
        captured = capsys.readouterr()
        # The live channel and the durable one are separate: a logger that
        # stopped reaching stderr would leave the footer intact.
        assert "does-not-exist" in captured.err
        assert captured.out.startswith("## Roadmap")
