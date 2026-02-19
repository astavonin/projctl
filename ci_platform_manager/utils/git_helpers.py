"""Git repository utilities."""

import subprocess
from pathlib import Path
from typing import Optional


def get_current_repo_path() -> Optional[str]:
    """Get current repository full path from git remote.

    Returns:
        Repository path (e.g., 'group/project') or None if not in a repo.
    """
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd()
        )

        remote_url = result.stdout.strip()

        # Parse GitLab/GitHub URL
        # Examples:
        #   https://gitlab.com/group/project.git
        #   git@gitlab.com:group/project.git
        #   https://github.com/owner/repo.git

        if '@' in remote_url:
            # SSH format: git@host:path.git
            path = remote_url.split(':', 1)[1]
        else:
            # HTTPS format: https://host/path.git
            path = remote_url.split('/', 3)[-1]

        # Remove .git suffix
        if path.endswith('.git'):
            path = path[:-4]

        return path

    except (subprocess.CalledProcessError, IndexError):
        return None
