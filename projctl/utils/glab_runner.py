"""Shared utility for running glab CLI commands."""

import logging
from typing import List

from .cli_runner import run_cli_command

logger = logging.getLogger(__name__)

_NOT_FOUND_MSG = "glab command not found. Please install glab CLI."


def run_glab_command(cmd: List[str]) -> str:
    """Run a glab command and return its output.

    Args:
        cmd: List of command arguments to pass to glab.

    Returns:
        Command output as a string.

    Raises:
        PlatformError: If the command fails or glab is not installed.
    """
    return run_cli_command("glab", cmd, _NOT_FOUND_MSG)
