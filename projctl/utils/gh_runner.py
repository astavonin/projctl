"""Shared utility for running gh CLI commands."""

import logging
from typing import List

from .cli_runner import run_cli_command

logger = logging.getLogger(__name__)

_NOT_FOUND_MSG = (
    "gh command not found. Please install GitHub CLI (https://cli.github.com)."
)


def run_gh_command(cmd: List[str]) -> str:
    """Run a gh command and return its output.

    Args:
        cmd: List of command arguments to pass to gh.

    Returns:
        Command output as a string.

    Raises:
        PlatformError: If the command fails or gh is not installed.
    """
    return run_cli_command("gh", cmd, _NOT_FOUND_MSG)
