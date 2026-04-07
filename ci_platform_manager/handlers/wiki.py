"""GitLab project wiki handler."""

import json
import logging
import urllib.parse
from typing import Any, Dict, List

from ..exceptions import PlatformError
from ..utils.git_helpers import get_current_repo_path, get_gitlab_base_url
from ..utils.glab_runner import run_glab_command

logger = logging.getLogger(__name__)


class WikiHandler:
    """Handler for GitLab project wiki operations."""

    def __init__(self) -> None:
        """Initialize the wiki handler using the current repository's project path."""
        project_path = get_current_repo_path()
        if not project_path:
            raise PlatformError(
                "Cannot determine project path. Run from a git repository with a GitLab remote."
            )
        self._project_path = project_path
        self._encoded_project = urllib.parse.quote(project_path, safe="")
        self._base_url = get_gitlab_base_url()

    def _page_url(self, slug: str) -> str:
        """Construct the web URL for a wiki page slug."""
        if self._base_url:
            return f"{self._base_url}/{self._project_path}/-/wikis/{slug}"
        return ""

    def list_pages(self) -> None:
        """List all wiki pages for the current project.

        Prints slug and title for each page.

        Raises:
            PlatformError: If the API call fails.
        """
        endpoint = f"projects/{self._encoded_project}/wikis"
        output = run_glab_command(["api", endpoint])
        pages: List[Dict[str, Any]] = json.loads(output) if output else []

        if not pages:
            print("No wiki pages found.")
            return

        for page in pages:
            slug = page.get("slug", "")
            title = page.get("title", "")
            print(f"{slug}\t{title}")

    def load_page(self, slug: str) -> None:
        """Load and print a single wiki page by slug.

        Args:
            slug: The wiki page slug.

        Raises:
            PlatformError: If the API call fails.
        """
        encoded_slug = urllib.parse.quote(slug, safe="")
        endpoint = f"projects/{self._encoded_project}/wikis/{encoded_slug}"
        output = run_glab_command(["api", endpoint])
        page: Dict[str, Any] = json.loads(output)

        print(f"# {page.get('title', '')}\n")
        print(f"**Slug:** {page.get('slug', '')}  ")
        print(f"**Format:** {page.get('format', '')}  ")
        print("\n## Content\n")
        print(page.get("content", ""))

    def create_page(self, title: str, content: str, dry_run: bool) -> None:
        """Create a new wiki page.

        Args:
            title: The page title.
            content: Markdown content for the page.
            dry_run: If True, print the payload without making API calls.

        Raises:
            PlatformError: If the API call fails.
        """
        if dry_run:
            print("[dry-run] Would POST to wiki with:")
            print(f"  title:   {title}")
            print("  format:  markdown")
            print(f"  content: {content[:120]}{'...' if len(content) > 120 else ''}")
            return

        endpoint = f"projects/{self._encoded_project}/wikis"
        # Build field args for glab api --field key=value
        args = [
            "api",
            "--method",
            "POST",
            endpoint,
            "--field",
            f"title={title}",
            "--field",
            f"content={content}",
            "--field",
            "format=markdown",
        ]
        output = run_glab_command(args)
        created: Dict[str, Any] = json.loads(output)

        slug = created.get("slug", "")
        url = created.get("web_url") or self._page_url(slug)
        print(f"Created wiki page: {slug}")
        print(f"URL: {url}")

    def _get_page_title(self, slug: str) -> str:
        """Fetch the current title for a wiki page.

        Args:
            slug: The wiki page slug.

        Returns:
            The page title, or the slug itself if the title cannot be fetched.
        """
        try:
            encoded_slug = urllib.parse.quote(slug, safe="")
            endpoint = f"projects/{self._encoded_project}/wikis/{encoded_slug}"
            output = run_glab_command(["api", endpoint])
            page: Dict[str, Any] = json.loads(output)
            return page.get("title") or slug
        except (PlatformError, json.JSONDecodeError, KeyError) as err:
            logger.warning("Could not fetch existing page title for '%s': %s", slug, err)
            return slug

    def update_page(self, slug: str, content: str, dry_run: bool) -> None:
        """Update an existing wiki page, preserving its current title.

        Args:
            slug: The wiki page slug to update.
            content: New Markdown content for the page.
            dry_run: If True, print the payload without making API calls.

        Raises:
            PlatformError: If the API call fails.
        """
        title = self._get_page_title(slug)

        if dry_run:
            print(f"[dry-run] Would PUT wiki page '{slug}' with:")
            print(f"  title:   {title}")
            print("  format:  markdown")
            print(f"  content: {content[:120]}{'...' if len(content) > 120 else ''}")
            return

        encoded_slug = urllib.parse.quote(slug, safe="")
        endpoint = f"projects/{self._encoded_project}/wikis/{encoded_slug}"
        args = [
            "api",
            "--method",
            "PUT",
            endpoint,
            "--field",
            f"title={title}",
            "--field",
            f"content={content}",
            "--field",
            "format=markdown",
        ]
        output = run_glab_command(args)
        updated: Dict[str, Any] = json.loads(output)

        updated_slug = updated.get("slug", slug)
        url = updated.get("web_url") or self._page_url(updated_slug)
        print(f"Updated wiki page: {updated_slug}")
        print(f"URL: {url}")
