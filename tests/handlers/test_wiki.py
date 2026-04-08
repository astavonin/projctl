"""Tests for projctl.handlers.wiki module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from projctl.exceptions import PlatformError
from projctl.handlers.wiki import WikiHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_PATH = "group/my-project"
_ENCODED_PROJECT = "group%2Fmy-project"

SAMPLE_PAGES = [
    {"slug": "home", "title": "Home"},
    {"slug": "getting-started", "title": "Getting Started"},
]

SAMPLE_PAGE = {
    "slug": "home",
    "title": "Home",
    "format": "markdown",
    "content": "# Home\n\nWelcome to the wiki.",
}


def _make_handler() -> WikiHandler:
    """Create a WikiHandler with a mocked git remote."""
    with patch(
        "projctl.handlers.wiki.get_current_repo_path",
        return_value=_PROJECT_PATH,
    ):
        return WikiHandler()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestWikiHandlerInit:
    """Test WikiHandler initialisation."""

    def test_init_success(self) -> None:
        """Handler stores project path and pre-encodes it."""
        handler = _make_handler()

        assert handler._project_path == _PROJECT_PATH
        assert handler._encoded_project == _ENCODED_PROJECT

    def test_init_no_remote_raises(self) -> None:
        """Missing git remote raises PlatformError."""
        with patch(
            "projctl.handlers.wiki.get_current_repo_path",
            return_value=None,
        ):
            with pytest.raises(PlatformError, match="Cannot determine project path"):
                WikiHandler()


# ---------------------------------------------------------------------------
# list_pages
# ---------------------------------------------------------------------------


class TestListPages:
    """Test WikiHandler.list_pages."""

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_list_pages_prints_slug_and_title(
        self, mock_run: object, capsys: pytest.CaptureFixture
    ) -> None:
        """Each page is printed as 'slug\\ttitle'."""
        mock_run.return_value = json.dumps(SAMPLE_PAGES)
        handler = _make_handler()

        handler.list_pages()

        out = capsys.readouterr().out
        assert "home\tHome" in out
        assert "getting-started\tGetting Started" in out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_list_pages_calls_correct_endpoint(self, mock_run: object) -> None:
        """The wikis list endpoint is called with the encoded project path."""
        mock_run.return_value = json.dumps([])
        handler = _make_handler()

        handler.list_pages()

        mock_run.assert_called_once_with(["api", f"projects/{_ENCODED_PROJECT}/wikis"])

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_list_pages_empty(self, mock_run: object, capsys: pytest.CaptureFixture) -> None:
        """Empty page list prints a helpful message."""
        mock_run.return_value = json.dumps([])
        handler = _make_handler()

        handler.list_pages()

        assert "No wiki pages found" in capsys.readouterr().out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_list_pages_propagates_platform_error(self, mock_run: object) -> None:
        """PlatformError from glab runner is propagated."""
        mock_run.side_effect = PlatformError("glab failed")
        handler = _make_handler()

        with pytest.raises(PlatformError):
            handler.list_pages()


# ---------------------------------------------------------------------------
# load_page
# ---------------------------------------------------------------------------


class TestLoadPage:
    """Test WikiHandler.load_page."""

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_load_page_prints_fields(self, mock_run: object, capsys: pytest.CaptureFixture) -> None:
        """Title, slug, format, and content are printed."""
        mock_run.return_value = json.dumps(SAMPLE_PAGE)
        handler = _make_handler()

        handler.load_page("home")

        out = capsys.readouterr().out
        assert "Home" in out
        assert "home" in out
        assert "markdown" in out
        assert "Welcome to the wiki" in out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_load_page_calls_correct_endpoint(self, mock_run: object) -> None:
        """The wikis/{slug} endpoint is called."""
        mock_run.return_value = json.dumps(SAMPLE_PAGE)
        handler = _make_handler()

        handler.load_page("home")

        mock_run.assert_called_once_with(["api", f"projects/{_ENCODED_PROJECT}/wikis/home"])

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_load_page_encodes_slug(self, mock_run: object) -> None:
        """Slugs with special characters are URL-encoded."""
        mock_run.return_value = json.dumps(
            {"slug": "my page", "title": "My Page", "format": "markdown", "content": ""}
        )
        handler = _make_handler()

        handler.load_page("my page")

        call_args = mock_run.call_args[0][0]
        assert "my%20page" in call_args[-1]


# ---------------------------------------------------------------------------
# create_page
# ---------------------------------------------------------------------------


class TestCreatePage:
    """Test WikiHandler.create_page."""

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_create_page_dry_run_does_not_call_api(self, mock_run: object) -> None:
        """Dry-run mode prints payload and skips the API call."""
        handler = _make_handler()

        handler.create_page(title="My Page", content="# Content", dry_run=True)

        mock_run.assert_not_called()

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_create_page_dry_run_output(
        self, mock_run: object, capsys: pytest.CaptureFixture
    ) -> None:
        """Dry-run output includes title and content preview."""
        handler = _make_handler()

        handler.create_page(title="My Page", content="# Content", dry_run=True)

        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "My Page" in out
        assert "# Content" in out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_create_page_posts_to_correct_endpoint(self, mock_run: object) -> None:
        """POST is sent to the wikis endpoint."""
        created = {"slug": "my-page", "web_url": "https://gitlab.example.com/wiki/my-page"}
        mock_run.return_value = json.dumps(created)
        handler = _make_handler()

        handler.create_page(title="My Page", content="# Content", dry_run=False)

        call_args = mock_run.call_args[0][0]
        assert "--method" in call_args
        assert "POST" in call_args
        assert f"projects/{_ENCODED_PROJECT}/wikis" in call_args

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_create_page_prints_slug_and_url(
        self, mock_run: object, capsys: pytest.CaptureFixture
    ) -> None:
        """Created page slug and URL are printed."""
        created = {"slug": "my-page", "web_url": "https://gitlab.example.com/wiki/my-page"}
        mock_run.return_value = json.dumps(created)
        handler = _make_handler()

        handler.create_page(title="My Page", content="# Content", dry_run=False)

        out = capsys.readouterr().out
        assert "my-page" in out
        assert "https://gitlab.example.com/wiki/my-page" in out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_create_page_includes_markdown_format(self, mock_run: object) -> None:
        """The format=markdown field is always sent."""
        created = {"slug": "new", "web_url": "http://example.com"}
        mock_run.return_value = json.dumps(created)
        handler = _make_handler()

        handler.create_page(title="New", content="body", dry_run=False)

        call_args = mock_run.call_args[0][0]
        assert "format=markdown" in call_args

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_create_page_content_truncated_in_dry_run(
        self, mock_run: object, capsys: pytest.CaptureFixture
    ) -> None:
        """Long content is truncated in dry-run preview."""
        long_content = "x" * 200
        handler = _make_handler()

        handler.create_page(title="T", content=long_content, dry_run=True)

        out = capsys.readouterr().out
        assert "..." in out


# ---------------------------------------------------------------------------
# update_page
# ---------------------------------------------------------------------------


class TestUpdatePage:
    """Test WikiHandler.update_page."""

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_update_page_dry_run_does_not_call_api(self, mock_run: object) -> None:
        """Dry-run mode fetches title but skips the PUT call."""
        mock_run.return_value = json.dumps(SAMPLE_PAGE)
        handler = _make_handler()

        handler.update_page(slug="home", content="# New Content", dry_run=True)

        # Only the GET for title should have been called, not PUT
        assert mock_run.call_count == 1
        call_args = mock_run.call_args[0][0]
        assert "--method" not in call_args

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_update_page_dry_run_output(
        self, mock_run: object, capsys: pytest.CaptureFixture
    ) -> None:
        """Dry-run output includes slug and content preview."""
        mock_run.return_value = json.dumps(SAMPLE_PAGE)
        handler = _make_handler()

        handler.update_page(slug="home", content="# New", dry_run=True)

        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "home" in out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_update_page_puts_to_correct_endpoint(self, mock_run: object) -> None:
        """PUT is sent to the wikis/{slug} endpoint."""
        updated = {"slug": "home", "web_url": "https://gitlab.example.com/wiki/home"}
        # First call: GET for title; second call: PUT
        mock_run.side_effect = [json.dumps(SAMPLE_PAGE), json.dumps(updated)]
        handler = _make_handler()

        handler.update_page(slug="home", content="# Updated", dry_run=False)

        put_call_args = mock_run.call_args_list[1][0][0]
        assert "--method" in put_call_args
        assert "PUT" in put_call_args
        assert f"projects/{_ENCODED_PROJECT}/wikis/home" in put_call_args

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_update_page_prints_slug_and_url(
        self, mock_run: object, capsys: pytest.CaptureFixture
    ) -> None:
        """Updated page slug and URL are printed."""
        updated = {"slug": "home", "web_url": "https://gitlab.example.com/wiki/home"}
        mock_run.side_effect = [json.dumps(SAMPLE_PAGE), json.dumps(updated)]
        handler = _make_handler()

        handler.update_page(slug="home", content="# Updated", dry_run=False)

        out = capsys.readouterr().out
        assert "home" in out
        assert "https://gitlab.example.com/wiki/home" in out

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_update_page_preserves_title(self, mock_run: object) -> None:
        """Existing page title is preserved in the PUT request."""
        updated = {"slug": "home", "web_url": "http://example.com"}
        mock_run.side_effect = [json.dumps(SAMPLE_PAGE), json.dumps(updated)]
        handler = _make_handler()

        handler.update_page(slug="home", content="# Updated", dry_run=False)

        put_call_args = mock_run.call_args_list[1][0][0]
        # title=Home should appear in the --field arguments
        assert "title=Home" in put_call_args

    @patch("projctl.handlers.wiki.run_glab_command")
    def test_update_page_falls_back_to_slug_on_title_fetch_error(self, mock_run: object) -> None:
        """If fetching the existing title fails, slug is used as fallback title."""
        updated = {"slug": "home", "web_url": "http://example.com"}
        mock_run.side_effect = [PlatformError("not found"), json.dumps(updated)]
        handler = _make_handler()

        # Should not raise
        handler.update_page(slug="home", content="# Updated", dry_run=False)

        put_call_args = mock_run.call_args_list[1][0][0]
        assert "title=home" in put_call_args


# ---------------------------------------------------------------------------
# cmd_wiki integration (CLI dispatch)
# ---------------------------------------------------------------------------


class TestCmdWiki:
    """Test cmd_wiki CLI dispatch."""

    def _make_args(self, wiki_command: str, **kwargs):
        """Build a minimal args namespace."""
        import argparse

        args = argparse.Namespace(wiki_command=wiki_command, **kwargs)
        return args

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_list(self, mock_handler_cls: object) -> None:
        """wiki list delegates to handler.list_pages."""
        from projctl.cli import cmd_wiki

        args = self._make_args("list")
        result = cmd_wiki(args)

        assert result == 0
        mock_handler_cls.return_value.list_pages.assert_called_once()

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_load(self, mock_handler_cls: object) -> None:
        """wiki load delegates to handler.load_page with slug."""
        from projctl.cli import cmd_wiki

        args = self._make_args("load", slug="home")
        result = cmd_wiki(args)

        assert result == 0
        mock_handler_cls.return_value.load_page.assert_called_once_with("home")

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_create(self, mock_handler_cls: object, tmp_path: Path) -> None:
        """wiki create reads the content file and delegates to handler.create_page."""
        from projctl.cli import cmd_wiki

        content_file = tmp_path / "page.md"
        content_file.write_text("# Hello")

        args = self._make_args("create", title="My Page", content=str(content_file), dry_run=False)
        result = cmd_wiki(args)

        assert result == 0
        mock_handler_cls.return_value.create_page.assert_called_once_with(
            title="My Page", content="# Hello", dry_run=False
        )

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_create_missing_file(self, mock_handler_cls: object, tmp_path: Path) -> None:
        """wiki create returns 1 when the content file does not exist."""
        from projctl.cli import cmd_wiki

        args = self._make_args(
            "create",
            title="My Page",
            content=str(tmp_path / "nonexistent.md"),
            dry_run=False,
        )
        result = cmd_wiki(args)

        assert result == 1
        mock_handler_cls.return_value.create_page.assert_not_called()

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_update(self, mock_handler_cls: object, tmp_path: Path) -> None:
        """wiki update reads the content file and delegates to handler.update_page."""
        from projctl.cli import cmd_wiki

        content_file = tmp_path / "updated.md"
        content_file.write_text("# Updated")

        args = self._make_args("update", slug="home", content=str(content_file), dry_run=True)
        result = cmd_wiki(args)

        assert result == 0
        mock_handler_cls.return_value.update_page.assert_called_once_with(
            slug="home", content="# Updated", dry_run=True
        )

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_update_missing_file(self, mock_handler_cls: object, tmp_path: Path) -> None:
        """wiki update returns 1 when the content file does not exist."""
        from projctl.cli import cmd_wiki

        args = self._make_args(
            "update",
            slug="home",
            content=str(tmp_path / "missing.md"),
            dry_run=False,
        )
        result = cmd_wiki(args)

        assert result == 1
        mock_handler_cls.return_value.update_page.assert_not_called()

    @patch("projctl.cli.WikiHandler")
    def test_cmd_wiki_platform_error_returns_1(self, mock_handler_cls: object) -> None:
        """PlatformError from handler is caught and returns exit code 1."""
        from projctl.cli import cmd_wiki

        mock_handler_cls.side_effect = PlatformError("no remote")

        args = self._make_args("list")
        result = cmd_wiki(args)

        assert result == 1
