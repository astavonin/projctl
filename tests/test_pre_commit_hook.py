"""Tests for scripts/pre-commit — the proprietary-identifier guard.

These run the real hook under a real ``bash`` against real throwaway git repositories.
The hook's whole job is to interpose on ``git commit``, and its two scan modes are built
from ``git grep --cached`` and ``git grep --untracked``; stubbing either boundary would
re-encode the assumption under test rather than check it. Hence integration, not unit.

Every subprocess runs with ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` pointed at
os.devnull. Without that the suite inherits the developer's git config, where a global
``core.hooksPath`` makes the installed-hook tests silently test nothing and
``commit.gpgsign`` fails the commits outright.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_SOURCE = REPO_ROOT / "scripts" / "pre-commit"
TEMPLATE_SOURCE = REPO_ROOT / "scripts" / "proprietary-terms.template"

pytestmark = pytest.mark.integration

# Kept out of the hook's own exemption path on purpose: the exemption is anchored to
# this module's real location, and test_exemption_is_anchored_to_this_files_real_path
# asserts the two agree, so a rename fails a test instead of the next commit.
THIS_FILE_REPO_PATH = "tests/test_pre_commit_hook.py"


def _env() -> dict:
    """Environment with git config isolation, so host config cannot alter results."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        # GIT_CONFIG_COUNT outranks GIT_CONFIG_GLOBAL, so isolating only the latter
        # leaves a hole; these three would point every fixture at the caller's repo.
        "GIT_CONFIG_COUNT": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for leak in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_TEMPLATE_DIR"):
        env.pop(leak, None)
    return env


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command inside repo, raising on failure."""
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=_env()
    )


def _git_ok(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command inside repo without raising."""
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False, env=_env()
    )


@pytest.fixture(name="repo")
def repo_fixture(tmp_path: Path) -> Path:
    """A throwaway git repo with the hook available at scripts/pre-commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy(HOOK_SOURCE, scripts / "pre-commit")
    (scripts / "pre-commit").chmod(0o755)
    shutil.copy(TEMPLATE_SOURCE, scripts / "proprietary-terms.template")
    return repo


def _run(repo: Path, *, staged: bool, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Invoke the hook directly, returning (exit code, stdout, stderr) separately.

    stdout and stderr are kept apart so a test can pin behaviour that only shows on
    one stream. The scanner is ``git grep``, not GNU grep: without ``-I`` it prints
    ``Binary file <path> matches`` on **stdout**, so ``out == ""`` is the assertion that
    pins that flag. ``err == ""`` is a separate catch-all against stray diagnostics —
    which now also means "the scan did not fail to read anything".

    staged=True sets GIT_INDEX_FILE, which is how the hook distinguishes a git-invoked
    run (scan the index) from a human-invoked sweep (scan the working tree).
    """
    env = _env()
    if staged:
        # Derive the index from the worktree we actually run in. Git honours
        # GIT_INDEX_FILE, so pointing it at the main worktree's index from inside a
        # linked worktree would make `git diff --cached` read the wrong index.
        git_dir = subprocess.run(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=cwd or repo,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.strip()
        env["GIT_INDEX_FILE"] = str(Path(git_dir) / "index")
    else:
        env.pop("GIT_INDEX_FILE", None)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "pre-commit")],
        cwd=cwd or repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _denylist(repo: Path, body: str) -> None:
    """Write the denylist into the repo's git common directory."""
    (repo / ".git" / "proprietary-terms").write_text(body, encoding="utf-8")


def _stage(repo: Path, name: str, content: str) -> None:
    """Write a file and stage it."""
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", "--", name)


def _seed_commit(repo: Path) -> None:
    """Create one commit so HEAD exists, bypassing the hook."""
    _stage(repo, "seed.py", "x = 1\n")
    _git(repo, "commit", "-q", "--no-verify", "-m", "seed")


class TestStructuralPattern:
    """The hardcoded work-path pattern, which needs no denylist to fire."""

    def test_clean_staged_file_passes(self, repo: Path) -> None:
        """A staged file with no proprietary identifier exits 0 and says nothing."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "clean.py", "x = 1\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0
        assert out == ""
        # stderr must be empty too: a diagnostic here would mean the scan errored and
        # was silently treated as clean, which is the fail-open this guard exists to avoid.
        assert err == ""

    def test_work_path_blocks_and_names_file_and_line(self, repo: Path) -> None:
        """A /home/<user>/work/ path exits 1 and reports the exact file and line."""
        _stage(repo, "leak.py", "# see /home/alice/work/secretproj/notes.md\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "leak.py:1:" in out
        assert "/home/alice/work/secretproj" in out

    @pytest.mark.parametrize(
        "path, blocked",
        [
            ("/home/alice/work/x", True),
            ("/home/bob-jr/work/x", True),
            ("/home/alice.smith/work/x", True),  # first.last is a standard account name
            ("/home/1alice/work/x", True),
            ("/home/a_b/work/x", True),
            ("/home/alice/projects/x", False),  # personal repo, not work context
            ("/home/alice/workspace/x", False),  # /work/ must be a whole segment
            ("/homeless/alice/work/x", False),  # a "homeless" dir is not a work path
        ],
    )
    def test_structural_pattern_boundaries(self, repo: Path, path: str, blocked: bool) -> None:
        """The username class and the /work/ segment are pinned at their boundaries."""
        _stage(repo, "f.py", f"{path}\n")

        code, out, _ = _run(repo, staged=True)

        assert code == (1 if blocked else 0), f"{path} -> {out}"

    def test_filename_is_reported_when_only_one_file_is_staged(self, repo: Path) -> None:
        """A single-file scan must still name the file, not print a bare line number."""
        _stage(repo, "only.py", "/home/bob/work/x\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "only.py:1:" in out

    def test_path_containing_a_space_is_scanned(self, repo: Path) -> None:
        """NUL-delimited collection must keep a spaced path as one argument."""
        _stage(repo, "my file.py", "/home/gina/work/q\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "my file.py:1:" in out

    def test_filename_that_looks_like_an_option_is_scanned(self, repo: Path) -> None:
        """A path beginning with '-' must not be parsed as a grep option."""
        _stage(repo, "--quiet", "/home/alice/work/secret/x\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "--quiet" in out


class TestIndexIsScannedNotWorktree:
    """The bytes that become history are the index's, not the working tree's."""

    def test_leak_staged_then_cleaned_in_worktree_is_still_blocked(self, repo: Path) -> None:
        """The central contract: what git will commit is what gets scanned.

        Reproduces the exact retry flow the hook's own message invites — edit the file,
        re-run git commit, forget to re-stage. Scanning the worktree let this through.
        """
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")
        (repo / "leak.py").write_text("totally clean\n", encoding="utf-8")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "leak.py:1:" in out

    def test_clean_index_with_dirty_worktree_is_not_blocked(self, repo: Path) -> None:
        """The inverse: unstaged content is not being committed, so it must not block."""
        _stage(repo, "f.py", "clean\n")
        (repo / "f.py").write_text("/home/alice/work/notyetstaged\n", encoding="utf-8")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out + err

    def test_staged_symlink_is_judged_by_its_own_blob(self, repo: Path) -> None:
        """git commits the link path text; the target's content is not in the commit."""
        (repo / "real.txt").write_text("/home/eve/work/secret/x\n", encoding="utf-8")
        os.symlink("real.txt", repo / "link.txt")
        _git(repo, "add", "--", "link.txt")

        code, out, err = _run(repo, staged=True)

        # The blob is the string "real.txt" — no work path — so this must not block.
        assert code == 0, out + err

    def test_symlink_whose_target_path_is_a_work_path_is_blocked(self, repo: Path) -> None:
        """When the link *text itself* is a work path, that text is what gets committed."""
        os.symlink("/home/dan/work/secretproj/config.yaml", repo / "cfg.yaml")
        _git(repo, "add", "--", "cfg.yaml")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "cfg.yaml" in out

    def test_typechange_to_a_leaking_symlink_is_scanned(self, repo: Path) -> None:
        """--diff-filter must include T: a typechange replaces the content wholesale."""
        _stage(repo, "t.txt", "harmless\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")
        (repo / "t.txt").unlink()
        os.symlink("/home/dan/work/secret/target.txt", repo / "t.txt")
        _git(repo, "add", "--", "t.txt")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "t.txt" in out


class TestDenylistedTerms:
    """Literal terms read from the git common dir, which git never publishes."""

    def test_denylisted_term_blocks(self, repo: Path) -> None:
        """A staged file containing a denylisted term exits 1 and names it."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "handler.py", "# verified against acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "handler.py:1:" in out
        assert "acmecorp.example" in out

    def test_term_matching_is_case_insensitive(self, repo: Path) -> None:
        """Casing must not be an escape hatch — the identifier is the same one."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "handler.py", "# host: ACMECorp.Example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "ACMECorp.Example" in out

    def test_term_is_matched_literally_not_as_a_regex(self, repo: Path) -> None:
        """Fixed-string matching is documented twice; -E would over-match here."""
        _denylist(repo, "acme.corp\n")
        _stage(repo, "a.py", "acmeXcorp is unrelated\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out + err

    def test_a_bracketed_term_is_matched_literally(self, repo: Path) -> None:
        """Under -E this term is an invalid regex, and the failure would be swallowed."""
        _denylist(repo, "acme[internal]\n")
        _stage(repo, "b.py", "the acme[internal] group\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "b.py:1:" in out

    def test_multi_word_term_matches_the_phrase_not_its_words(self, repo: Path) -> None:
        """Word-splitting the denylist would make each word its own term."""
        _denylist(repo, "Big Client Ltd\n")
        _stage(repo, "hit.py", "reviewed by big client ltd\n")

        code, out, _ = _run(repo, staged=True)
        assert code == 1
        assert "hit.py:1:" in out

        _git(repo, "rm", "-q", "--cached", "--", "hit.py")
        (repo / "hit.py").unlink()
        # Every word present, the phrase absent — must NOT block.
        _stage(repo, "miss.py", "Big improvements for the Client of Ltd\n")

        code, out, err = _run(repo, staged=True)
        assert code == 0, out + err

    def test_trailing_comment_is_stripped_from_a_term(self, repo: Path) -> None:
        """The template teaches this form; unstripped it becomes a term matching nothing."""
        _denylist(repo, "acmecorp.example    # internal hostname\n")
        _stage(repo, "c.py", "host = acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "c.py:1:" in out

    def test_whole_line_comment_does_not_become_a_term(self, repo: Path) -> None:
        """Without stripping, '# TODO' would match every file carrying that comment."""
        _denylist(repo, "# TODO\nacmecorp.example\n")
        _stage(repo, "d.py", "# TODO: refactor this\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out + err

    def test_a_term_beginning_with_a_hash_is_not_discarded(self, repo: Path) -> None:
        """A leading-# term is a real identifier shape (a channel or tag), not a comment."""
        _denylist(repo, "  #acme-secret-tag\n")
        _stage(repo, "e.py", "see #acme-secret-tag for details\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "e.py:1:" in out

    def test_absent_denylist_warns_and_still_runs_the_structural_pass(self, repo: Path) -> None:
        """An inert half must announce itself; the other half must keep working."""
        assert not (repo / ".git" / "proprietary-terms").exists()
        _stage(repo, "leak.py", "/home/bob/work/thing\n")

        code, out, err = _run(repo, staged=True)

        assert code == 1
        assert "leak.py:1:" in out
        assert "literal-term checking is OFF" in err

    def test_freshly_seeded_template_warns_that_checking_is_off(self, repo: Path) -> None:
        """This is the state right after `make install-hooks` — it must not look armed."""
        shutil.copy(TEMPLATE_SOURCE, repo / ".git" / "proprietary-terms")
        _stage(repo, "cfg.py", 'HOST = "gitlab.acme-internal.example"\n')

        code, out, err = _run(repo, staged=True)

        assert code == 0, out
        assert "literal-term checking is OFF" in err

    def test_zero_byte_denylist_does_not_match_every_line(self, repo: Path) -> None:
        """An empty pattern file would make grep match everything."""
        _denylist(repo, "")
        _stage(repo, "clean.py", "x = 1\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out
        assert "literal-term checking is OFF" in err

    def test_comment_only_denylist_does_not_match_every_line(self, repo: Path) -> None:
        """Distinct from the zero-byte case: lines exist but none yields a term."""
        _denylist(repo, "# nothing configured yet\n\n   \n")
        _stage(repo, "clean.py", "x = 1\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out
        assert "literal-term checking is OFF" in err


class TestFileSelection:
    """Which bytes each mode scans — the property that decides what escapes."""

    def test_sweep_scans_tracked_files_when_not_invoked_by_git(self, repo: Path) -> None:
        """With no GIT_INDEX_FILE the hook sweeps the working tree."""
        _stage(repo, "tracked.py", "/home/carol/work/x\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")

        code, out, _ = _run(repo, staged=False)

        assert code == 1
        assert "tracked.py:1:" in out
        assert "working tree" in out

    def test_sweep_scans_untracked_files(self, repo: Path) -> None:
        """A brand-new file is the one most likely to carry an unreviewed identifier."""
        _seed_commit(repo)
        (repo / "brand-new.py").write_text("/home/dave/work/y\n", encoding="utf-8")

        code, out, _ = _run(repo, staged=False)

        assert code == 1
        assert "brand-new.py:1:" in out

    def test_sweep_from_a_subdirectory_scans_the_whole_tree(self, repo: Path) -> None:
        """Reported paths must be root-relative, or the sweep silently scans nothing."""
        _stage(repo, "sub/dir/leaky.py", "/home/frank/work/internal/notes\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")

        code, out, _ = _run(repo, staged=False, cwd=repo / "sub" / "dir")

        assert code == 1
        assert "sub/dir/leaky.py:1:" in out

    def test_denylist_resolves_when_run_from_a_subdirectory(self, repo: Path) -> None:
        """--git-common-dir answers relative to CWD; resolving it wrong disables the half."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "sub/deep/leak.py", "host = acmecorp.example\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")

        code, out, _ = _run(repo, staged=False, cwd=repo / "sub" / "deep")

        assert code == 1
        assert "sub/deep/leak.py:1:" in out

    def test_denylist_resolves_from_a_linked_worktree(self, repo: Path, tmp_path: Path) -> None:
        """--git-common-dir over --git-dir exists precisely for this case."""
        _denylist(repo, "acmecorp.example\n")
        _seed_commit(repo)
        linked = tmp_path / "linked"
        _git(repo, "worktree", "add", "-q", "-b", "wt", str(linked))
        (linked / "a.py").write_text("host acmecorp.example\n", encoding="utf-8")
        _git(linked, "add", "--", "a.py")

        code, out, _ = _run(repo, staged=True, cwd=linked)

        assert code == 1
        assert "a.py:1:" in out

    def test_gitignored_file_is_not_swept(self, repo: Path) -> None:
        """An ignored file is never published, so flagging it would be a false block."""
        _stage(repo, ".gitignore", "ignored-artifacts/\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")
        (repo / "ignored-artifacts").mkdir()
        (repo / "ignored-artifacts" / "n.md").write_text("/home/eve/work/z\n", encoding="utf-8")

        code, out, err = _run(repo, staged=False)

        assert code == 0, out + err

    def test_force_added_ignored_file_is_still_scanned_at_commit_time(self, repo: Path) -> None:
        """Staging an ignored file publishes it, so the ignore no longer protects it."""
        _stage(repo, ".gitignore", "ignored-artifacts/\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")
        (repo / "ignored-artifacts").mkdir()
        (repo / "ignored-artifacts" / "n.md").write_text("/home/eve/work/z\n", encoding="utf-8")
        _git(repo, "add", "-f", "--", "ignored-artifacts/n.md")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "ignored-artifacts/n.md:1:" in out

    def test_binary_file_produces_no_report_on_either_stream(self, repo: Path) -> None:
        """-I keeps a build artifact out of the report AND out of the diagnostics.

        ``out == ""`` is what pins ``-I``: without it, ``git grep`` writes
        ``Binary file blob.bin matches`` to **stdout**, which lands in the report.
        ``err == ""`` separately pins that nothing failed to read.
        """
        _denylist(repo, "acmecorp.example\n")
        (repo / "blob.bin").write_bytes(b"\x00\x01/home/frank/work/bin\x00")
        _git(repo, "add", "--", "blob.bin")

        code, out, err = _run(repo, staged=True)

        assert code == 0
        assert out == ""
        assert err == ""

    def test_binary_file_does_not_suppress_a_sibling_text_hit(self, repo: Path) -> None:
        """Skipping the binary must not abort the scan of everything else."""
        (repo / "blob.bin").write_bytes(b"\x00\x01/home/frank/work/bin\x00")
        _git(repo, "add", "--", "blob.bin")
        _stage(repo, "text.py", "/home/frank/work/other\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "text.py:1:" in out
        assert "blob.bin" not in out

    def test_binary_file_holding_a_denylisted_term_is_not_reported(self, repo: Path) -> None:
        """-I must suppress binaries on the denylist pass too, not just the structural one."""
        _denylist(repo, "acmecorp.example\n")
        (repo / "blob.bin").write_bytes(b"\x00\x01acmecorp.example\x00")
        _git(repo, "add", "--", "blob.bin")

        code, out, err = _run(repo, staged=True)

        assert code == 0
        assert out == ""
        assert err == ""

    def test_staged_rename_of_a_leaking_file_is_scanned(self, repo: Path) -> None:
        """--diff-filter admits R; a rename still introduces the content at a new path."""
        _stage(repo, "old.py", "/home/carol/work/x\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")
        _git(repo, "mv", "old.py", "new.py")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "new.py:1:" in out

    def test_index_only_deletion_is_not_scanned(self, repo: Path) -> None:
        """A deletion removes content; there is nothing to scan and nothing to block."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "gone.py", "/home/carol/work/x\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")
        _git(repo, "rm", "-q", "--", "gone.py")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out + err
        assert err == ""

    def test_staged_file_removed_from_disk_does_not_error(self, repo: Path) -> None:
        """The index still holds the blob, so the scan proceeds from the index."""
        _stage(repo, "vanish.py", "/home/carol/work/x\n")
        (repo / "vanish.py").unlink()

        code, out, _ = _run(repo, staged=True)

        # The blob is still what would be committed, so it must still be caught.
        assert code == 1
        assert "vanish.py:1:" in out


class TestSelfExemption:
    """The one exempt path — a hole in the guard, so its scope is pinned here."""

    def test_hooks_own_test_file_is_exempt_from_the_structural_pass(self, repo: Path) -> None:
        """This very file needs work-path fixtures; without the exemption it self-blocks."""
        _stage(
            repo,
            THIS_FILE_REPO_PATH,
            '_stage(repo, "leak.py", "/home/alice/work/secretproj/x")\n',
        )

        code, out, err = _run(repo, staged=True)

        assert code == 0, out + err

    def test_exemption_does_not_extend_to_sibling_test_files(self, repo: Path) -> None:
        """The pattern is an anchored exact path — a directory-wide hole would be silent."""
        _stage(repo, "tests/test_something_else.py", "/home/alice/work/secretproj/x\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "tests/test_something_else.py:1:" in out

    def test_exemption_does_not_match_a_same_named_file_elsewhere(self, repo: Path) -> None:
        """Anchoring means one path, not any file with that basename."""
        _stage(repo, "vendor/tests/test_pre_commit_hook.py", "/home/alice/work/x\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "vendor/tests/test_pre_commit_hook.py:1:" in out

    def test_exemption_skips_only_that_file_and_the_scan_continues(self, repo: Path) -> None:
        """Skipping the file must not abandon the rest of the scan."""
        _stage(repo, THIS_FILE_REPO_PATH, "/home/alice/work/fixture\n")
        _stage(repo, "zz_leak.py", "/home/alice/work/secret/leak\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "zz_leak.py:1:" in out
        assert "test_pre_commit_hook.py" not in out

    def test_exemption_does_not_cover_the_denylist_pass(self, repo: Path) -> None:
        """Fixtures need work-paths, never a real proprietary term."""
        _denylist(repo, "acmecorp.internal\n")
        _stage(repo, THIS_FILE_REPO_PATH, "host = acmecorp.internal\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "test_pre_commit_hook.py" in out

    def test_exemption_also_applies_to_the_working_tree_sweep(self, repo: Path) -> None:
        """The recorded failure happened in sweep mode, so pin the exemption there too."""
        _stage(repo, THIS_FILE_REPO_PATH, "/home/alice/work/fixture\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "init")

        code, out, err = _run(repo, staged=False)

        assert code == 0, out + err

    def test_exemption_is_anchored_to_this_files_real_path(self) -> None:
        """A rename of this module would silently reproduce the recorded failure."""
        actual = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
        assert actual == THIS_FILE_REPO_PATH
        hook_text = HOOK_SOURCE.read_text(encoding="utf-8")
        assert THIS_FILE_REPO_PATH in hook_text


class TestScanFailuresAreLoud:
    """A guard that cannot tell 'found nothing' from 'could not look' fails open."""

    def test_unwritable_tmpdir_blocks_rather_than_skipping_the_denylist(self, repo: Path) -> None:
        """bare mktemp also fails unconditionally on macOS, so this is not exotic."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "l.py", "host = acmecorp.example\n")

        env = _env()
        env["GIT_INDEX_FILE"] = str(repo / ".git" / "index")
        env["TMPDIR"] = "/proc/nonexistent-xyz"
        result = subprocess.run(
            ["bash", str(repo / "scripts" / "pre-commit")],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert result.returncode == 1
        assert "temporary directory" in result.stderr
        # It must abort *there*. A downgrade of that fail() to a warning still exits 1,
        # but only after stumbling through later stages — which this pins out.
        assert "cannot write the denylist pattern file" not in result.stderr
        assert "Permission denied" not in result.stderr

    # Report formatting, not a scan failure — kept here only because it needs the same
    # two-rule fixture. See TestReportAttribution for the attribution assertions.
    def test_both_scans_hitting_prints_one_header(self, repo: Path) -> None:
        """Duplicate headers make one problem look like two."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "s.py", "/home/bob/work/thing\n")
        _stage(repo, "d.py", "host acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert out.count("proprietary identifiers found") == 1
        assert "s.py:1:" in out
        assert "d.py:1:" in out
        # The full label prefix — the bare phrase also appears in the epilogue, so
        # asserting on it would be satisfied by two independent sources.
        assert "work-path pattern (hardcoded in" in out
        assert "denylisted term (" in out
        # The remediation surface is the guard's whole supportability story.
        assert "Add a denylist term:" in out


class TestInstalledAsGitHook:
    """End to end: the hook's entire purpose is to interpose on `git commit`."""

    def _install(self, repo: Path) -> subprocess.CompletedProcess:
        """Run the real `make install-hooks`, not a stand-in for it."""
        shutil.copy(REPO_ROOT / "Makefile", repo / "Makefile")
        return subprocess.run(
            ["make", "install-hooks"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            env=_env(),
        )

    def _commit(self, repo: Path, message: str) -> subprocess.CompletedProcess:
        """Attempt a commit without raising, so the exit code can be asserted."""
        return _git_ok(repo, "commit", "-m", message)

    def test_commit_is_blocked_and_head_does_not_move(self, repo: Path) -> None:
        """The commit must not be created."""
        self._install(repo)
        _seed_commit(repo)
        before = _git(repo, "rev-parse", "HEAD").stdout.strip()
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")

        result = self._commit(repo, "should be blocked")

        assert result.returncode != 0
        assert "leak.py:1:" in result.stdout + result.stderr
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before

    def test_commit_succeeds_once_the_identifier_is_replaced(self, repo: Path) -> None:
        """After sanitising and re-staging, the same commit goes through."""
        self._install(repo)
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")
        assert self._commit(repo, "blocked").returncode != 0

        _stage(repo, "leak.py", "/home/alice/projects/myproject/x\n")
        result = self._commit(repo, "clean")

        assert result.returncode == 0, result.stdout + result.stderr

    def test_editing_without_restaging_does_not_let_the_leak_through(self, repo: Path) -> None:
        """The hook's own message invites this retry; it must not be a bypass."""
        self._install(repo)
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")
        assert self._commit(repo, "blocked").returncode != 0

        # Edit the file but forget to `git add` it — the index still holds the leak.
        (repo / "leak.py").write_text("clean now\n", encoding="utf-8")
        result = self._commit(repo, "second attempt")

        assert result.returncode != 0, "the staged leak was committed"
        assert "leak.py:1:" in result.stdout + result.stderr
        rev = _git_ok(repo, "rev-parse", "--verify", "HEAD")
        assert rev.returncode != 0, "a blocked commit must not have been created"

    def test_message_only_amend_is_not_blocked_by_untracked_files(self, repo: Path) -> None:
        """This project mandates --amend for fixes, so a false block here is on the hot path."""
        self._install(repo)
        _seed_commit(repo)
        (repo / "scratch.md").write_text("/home/bob/work/oldproject/todo\n", encoding="utf-8")

        result = _git_ok(repo, "commit", "--amend", "-m", "reworded")

        assert result.returncode == 0, result.stdout + result.stderr

    def test_deletion_only_commit_is_not_blocked_by_untracked_files(self, repo: Path) -> None:
        """An empty ACMRT list must not be mistaken for a manual sweep."""
        self._install(repo)
        _seed_commit(repo)
        (repo / "scratch.md").write_text("/home/bob/work/oldproject/todo\n", encoding="utf-8")
        _git(repo, "rm", "-q", "--", "seed.py")

        result = self._commit(repo, "delete only")

        assert result.returncode == 0, result.stdout + result.stderr

    def test_install_preserves_a_pre_existing_hook(self, repo: Path) -> None:
        """.git/hooks is unversioned, so a clobbered hook is unrecoverable."""
        hooks = repo / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "pre-commit").write_text("#!/bin/sh\necho MINE\n", encoding="utf-8")
        (hooks / "pre-commit").chmod(0o755)

        out = self._install(repo).stdout

        assert "Preserved" in out
        assert (hooks / "pre-commit.local.bak").read_text(encoding="utf-8") == (
            "#!/bin/sh\necho MINE\n"
        )

    def test_reinstall_does_not_back_up_its_own_shim(self, repo: Path) -> None:
        """Idempotence: repeated `make install` must not accumulate backups."""
        self._install(repo)
        self._install(repo)

        assert not (repo / ".git" / "hooks" / "pre-commit.local.bak").exists()

    def test_install_seeds_the_denylist_but_never_overwrites_it(self, repo: Path) -> None:
        """The denylist is user data with no backup anywhere."""
        self._install(repo)
        denylist = repo / ".git" / "proprietary-terms"
        assert denylist.exists()

        denylist.write_text("my-own-term\n", encoding="utf-8")
        self._install(repo)

        assert "my-own-term" in denylist.read_text(encoding="utf-8")

    def test_missing_tracked_script_fails_closed(self, repo: Path) -> None:
        """A checkout predating the guard, or a stray chmod -x, must not disarm it."""
        self._install(repo)
        _seed_commit(repo)
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")
        before = _git(repo, "rev-parse", "HEAD").stdout.strip()
        (repo / "scripts" / "pre-commit").unlink()

        result = self._commit(repo, "should be blocked")

        assert result.returncode != 0
        assert "refusing to commit" in result.stdout + result.stderr
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before

    def test_non_executable_tracked_script_fails_closed(self, repo: Path) -> None:
        """The same disarm, one chmod away."""
        self._install(repo)
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")
        (repo / "scripts" / "pre-commit").chmod(0o644)

        result = self._commit(repo, "should be blocked")

        assert result.returncode != 0
        assert "refusing to commit" in result.stdout + result.stderr

    def test_a_second_foreign_hook_does_not_destroy_the_first_backup(self, repo: Path) -> None:
        """.git/hooks is unversioned, so an overwritten backup is unrecoverable."""
        hooks = repo / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "pre-commit").write_text("#!/bin/sh\necho HOOK-A\n", encoding="utf-8")
        self._install(repo)
        (hooks / "pre-commit").write_text("#!/bin/sh\necho HOOK-B\n", encoding="utf-8")
        self._install(repo)

        preserved = "".join(
            f.read_text(encoding="utf-8") for f in hooks.glob("pre-commit.local.bak*")
        )
        assert "HOOK-A" in preserved
        assert "HOOK-B" in preserved

    def test_seeded_denylist_is_not_world_readable(self, repo: Path) -> None:
        """It holds exactly the strings that must not be disclosed."""
        self._install(repo)

        mode = (repo / ".git" / "proprietary-terms").stat().st_mode & 0o777
        assert mode == 0o600, oct(mode)

    def test_commit_only_a_leaking_path_is_blocked(self, repo: Path) -> None:
        """`git commit --only` builds a different index file than .git/index."""
        self._install(repo)
        _seed_commit(repo)
        _stage(repo, "leak.py", "/home/alice/work/secretproj/x\n")
        before = _git(repo, "rev-parse", "HEAD").stdout.strip()

        result = _git_ok(repo, "commit", "--only", "leak.py", "-m", "only")

        assert result.returncode != 0
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == before

    def test_commit_all_catches_an_unstaged_edit(self, repo: Path) -> None:
        """`git commit -a` stages into a lock index before the hook runs."""
        self._install(repo)
        _stage(repo, "f.py", "clean\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")
        (repo / "f.py").write_text("/home/alice/work/secretproj/x\n", encoding="utf-8")

        result = _git_ok(repo, "commit", "-a", "-m", "all")

        assert result.returncode != 0

    def test_updating_the_tracked_script_takes_effect_without_reinstalling(
        self, repo: Path
    ) -> None:
        """A copied hook would keep running the old ruleset after a pull."""
        self._install(repo)
        _seed_commit(repo)
        (repo / "scripts" / "pre-commit").write_text(
            "#!/usr/bin/env bash\necho UPDATED-RULESET >&2\nexit 1\n", encoding="utf-8"
        )
        _stage(repo, "anything.py", "x = 1\n")

        result = self._commit(repo, "should hit the updated script")

        assert result.returncode != 0
        assert "UPDATED-RULESET" in result.stdout + result.stderr


class TestPathspecMagicInFilenames:
    """Collected paths are git PATHSPECS, so a ':'-prefixed name is not inert data."""

    def test_a_file_named_like_exclude_magic_does_not_unguard_its_sibling(self, repo: Path) -> None:
        """`:^handler.py` is git's exclude short-magic; unprotected it removes handler.py."""
        _stage(repo, "handler.py", "/home/alice/work/secretproj/creds\n")
        _stage(repo, ":^handler.py", "")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "handler.py:1:" in out

    def test_a_file_whose_own_name_is_pathspec_magic_is_still_scanned(self, repo: Path) -> None:
        """Without :(literal) the odd file is selected by nothing and skipped silently."""
        (repo / ":config.py").write_text("/home/alice/work/secret/formA\n", encoding="utf-8")
        # `git add -- :config.py` would itself parse the name as pathspec magic.
        _git(repo, "add", "--", "./:config.py")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "config.py" in out


class TestUnscannablePathsBlock:
    """A path git refuses to scan is not a clean path."""

    def test_gitattributes_marking_a_text_path_binary_blocks(self, repo: Path) -> None:
        """`-diff` makes `git grep -I` skip the glob; unreported that is a silent hole."""
        _stage(repo, ".gitattributes", "*.md -diff\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "attrs")
        _stage(repo, "notes.md", "/home/alice/work/secret/viaattr\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "cannot be scanned" in out
        assert "notes.md" in out

    def test_the_binary_macro_blocks_the_same_way(self, repo: Path) -> None:
        """`binary` is a macro that unsets diff, so it must be caught identically."""
        _stage(repo, ".gitattributes", "*.md binary\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "attrs")
        _stage(repo, "n.md", "/home/alice/work/secret/viamacro\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "cannot be scanned" in out

    def test_an_ordinary_gitattributes_entry_does_not_block(self, repo: Path) -> None:
        """Only an unset diff attribute is unscannable; `text` and friends are fine."""
        _stage(repo, ".gitattributes", "*.md text\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "attrs")
        _stage(repo, "n.md", "hello\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0, out + err

    def test_an_unreadable_file_blocks_instead_of_reading_as_clean(self, repo: Path) -> None:
        """git grep reports this on stderr while still exiting 1 — the exit code lies."""
        _denylist(repo, "acmecorp.internal\n")
        _stage(repo, "secret.md", "/home/alice/work/secretproj/x\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")
        (repo / "secret.md").chmod(0o000)
        try:
            code, out, err = _run(repo, staged=False)
        finally:
            (repo / "secret.md").chmod(0o644)

        assert code == 1
        assert "could not read" in err
        assert "working tree clean" not in out


class TestSweepChecksSymlinks:
    """The sweep audits what is already in the tree — including links."""

    def test_sweep_reports_a_committed_leaking_symlink(self, repo: Path) -> None:
        """A link committed under --no-verify is exactly the sweep's target case."""
        os.symlink("/home/dan/work/secretproj/config.yaml", repo / "cfg.yaml")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")

        code, out, _ = _run(repo, staged=False)

        assert code == 1
        assert "cfg.yaml" in out
        assert "symlink target path" in out
        assert "working tree clean" not in out

    def test_symlink_whose_target_carries_a_denylisted_term_is_blocked(self, repo: Path) -> None:
        """The denylist half of the symlink pass, which nothing pinned before."""
        _denylist(repo, "acmecorp.example\n")
        os.symlink("//acmecorp.example/share/notes.md", repo / "linky")
        _git(repo, "add", "-A")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "linky" in out


class TestReportAttribution:
    """A hit must be attributed to the rule that actually fired."""

    def test_a_structural_hit_is_not_also_reported_as_a_denylisted_term(self, repo: Path) -> None:
        """Mis-attribution sends the developer to edit the wrong place."""
        _denylist(repo, "a-term-that-does-not-appear\n")
        _stage(repo, "s.py", "/home/alice/work/secretproj/x\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "work-path pattern" in out
        assert "denylisted term" not in out

    def test_a_denylist_hit_is_not_also_reported_as_a_work_path(self, repo: Path) -> None:
        """The mirror direction."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "d.py", "host acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "denylisted term" in out
        # The block label, not the bare phrase — the remediation footer mentions it too.
        assert "work-path pattern (hardcoded" not in out


class TestCleanSweepAnnouncesItself:
    """Silence is what a sweep that scanned nothing looks like — say something."""

    def test_a_clean_sweep_reports_the_active_term_count(self, repo: Path) -> None:
        """The count is the part that distinguishes 'armed' from 'inert'."""
        _denylist(repo, "one.example\ntwo.example\nthree.example\n")
        _stage(repo, "ok.py", "x = 1\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")

        code, out, _ = _run(repo, staged=False)

        assert code == 0
        assert "working tree clean (3 denylist term(s) active)" in out

    def test_a_clean_commit_stays_silent_on_stdout(self, repo: Path) -> None:
        """The confirmation is for the sweep only; a hook must not chatter."""
        _denylist(repo, "one.example\n")
        _stage(repo, "ok.py", "x = 1\n")

        code, out, err = _run(repo, staged=True)

        assert code == 0
        assert out == ""
        assert err == ""


class TestDenylistParsingEdgeCases:
    """Forms the template documents, which the parser must actually honour."""

    def test_a_hash_immediately_after_a_term_still_opens_a_comment(self, repo: Path) -> None:
        """The rule is '# followed by whitespace', not '# preceded by whitespace'."""
        _denylist(repo, "acmecorp.example# internal hostname\n")
        _stage(repo, "c.py", "host = acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "c.py:1:" in out

    def test_an_unreadable_denylist_blocks_rather_than_reading_as_empty(self, repo: Path) -> None:
        """A denylist that exists but cannot be read is a failure to look, not zero terms."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "l.py", "host = acmecorp.example\n")
        (repo / ".git" / "proprietary-terms").chmod(0o000)
        try:
            code, out, err = _run(repo, staged=True)
        finally:
            (repo / ".git" / "proprietary-terms").chmod(0o600)

        assert code == 1
        assert "cannot be read" in err
        assert "checking is OFF" not in err

    def test_a_utf8_bom_does_not_void_the_first_term(self, repo: Path) -> None:
        """A term pasted from a browser or Windows editor often carries one."""
        (repo / ".git" / "proprietary-terms").write_bytes(b"\xef\xbb\xbfacmecorp.example\n")
        _stage(repo, "b.py", "host = acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "b.py:1:" in out

    def test_a_crlf_denylist_still_matches(self, repo: Path) -> None:
        """\r is stripped as trailing whitespace; without that the term never matches."""
        (repo / ".git" / "proprietary-terms").write_bytes(b"acmecorp.example\r\n")
        _stage(repo, "r.py", "host = acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "r.py:1:" in out


class TestTemporaryStateIsCleanedUp:
    """The temp dir holds the decommented denylist — the terms themselves."""

    def test_no_workdir_survives_a_normal_run(self, repo: Path, tmp_path: Path) -> None:
        """Leaving proprietary terms in TMPDIR after every commit is the failure."""
        scratch = tmp_path / "tmp-probe"
        scratch.mkdir()
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "ok.py", "x = 1\n")

        env = _env()
        env["GIT_INDEX_FILE"] = str(repo / ".git" / "index")
        env["TMPDIR"] = str(scratch)
        result = subprocess.run(
            ["bash", str(repo / "scripts" / "pre-commit")],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert result.returncode == 0
        assert list(scratch.glob("projctl-precommit.*")) == []


class TestRemainingSelectionShapes:
    """Shapes the diff-filter and the chunk loop admit but nothing exercised."""

    def test_a_modification_to_a_tracked_file_is_scanned(self, repo: Path) -> None:
        """The commonest commit shape: edit a committed file, stage the edit."""
        _stage(repo, "cfg.py", "clean = True\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")
        _stage(repo, "cfg.py", "clean = True\n# /home/alice/work/secretproj/notes\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "cfg.py:2:" in out

    def test_every_path_is_scanned_across_chunk_boundaries(self, repo: Path) -> None:
        """The scan chunks at 400 paths; a stride or slice drift drops files silently.

        Leaks are spread across the whole range and every one is asserted: a single
        leak would land in a chunk a mutated stride still happens to cover, which is
        how the first version of this test passed against a broken stride.
        """
        leak_indices = {0, 199, 399, 400, 401, 599, 799}
        for i in range(800):
            body = f"/home/alice/work/secret/at{i:03d}\n" if i in leak_indices else "clean\n"
            (repo / f"f{i:03d}.txt").write_text(body, encoding="utf-8")
        _git(repo, "add", "-A")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        missing = [i for i in sorted(leak_indices) if f"f{i:03d}.txt:1:" not in out]
        assert not missing, f"chunking dropped: {missing}"

    def test_a_dash_prefixed_filename_is_scanned_by_the_denylist_pass_too(self, repo: Path) -> None:
        """`--` was pinned on the structural pass only; the terms pass needs it as well."""
        _denylist(repo, "acmecorp.example\n")
        _stage(repo, "--quiet", "host = acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "--quiet" in out

    def test_exemption_does_not_cover_the_denylist_pass_in_sweep_mode(self, repo: Path) -> None:
        """The structural half got a sweep twin; the denylist half needs one too."""
        _denylist(repo, "acmecorp.internal\n")
        _stage(repo, THIS_FILE_REPO_PATH, "host = acmecorp.internal\n")
        _git(repo, "commit", "-q", "--no-verify", "-m", "seed")

        code, out, _ = _run(repo, staged=False)

        assert code == 1
        assert "test_pre_commit_hook.py" in out
        assert "working tree clean" not in out

    def test_running_outside_a_git_worktree_aborts(self, tmp_path: Path) -> None:
        """Without this the hook would cd to an empty root and scan whatever it finds."""
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        shutil.copy(HOOK_SOURCE, outside / "pre-commit")
        (outside / "pre-commit").chmod(0o755)

        result = subprocess.run(
            ["bash", str(outside / "pre-commit")],
            cwd=outside,
            capture_output=True,
            text=True,
            check=False,
            env=_env(),
        )

        assert result.returncode == 1
        # It must abort *there*, on that one line. Downgrading the abort to a warning
        # still exits 1 — `cd ""` succeeds, so execution runs on to a later, different
        # failure — so the exit code and the message alone do not pin it.
        # git's own "fatal: not a git repository" also lands on stderr, hence the filter.
        ours = [ln for ln in result.stderr.splitlines() if ln.startswith("pre-commit:")]
        assert ours == ["pre-commit: not inside a git working tree"], ours


class TestDenylistLineHandling:
    """Line-level parsing the trim and the final-line guard are responsible for."""

    def test_a_term_with_trailing_whitespace_still_matches(self, repo: Path) -> None:
        """Without the trim the term carries spaces and silently matches nothing."""
        (repo / ".git" / "proprietary-terms").write_text("acmecorp.example   \n", encoding="utf-8")
        _stage(repo, "w.py", "host = acmecorp.example\n")

        code, out, _ = _run(repo, staged=True)

        assert code == 1
        assert "w.py:1:" in out

    def test_a_denylist_without_a_trailing_newline_still_yields_its_term(self, repo: Path) -> None:
        """A hand-edited file often lacks one; dropping the term would be silent."""
        (repo / ".git" / "proprietary-terms").write_text("acmecorp.example", encoding="utf-8")
        _stage(repo, "n.py", "host = acmecorp.example\n")

        code, out, err = _run(repo, staged=True)

        assert code == 1
        assert "n.py:1:" in out
        assert "checking is OFF" not in err
