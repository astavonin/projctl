"""Shared subprocess execution for CLI tools (gh, glab, etc.)."""

import logging
import subprocess
from typing import List

from ..exceptions import PlatformError

logger = logging.getLogger(__name__)


def run_cli_command(cli_name: str, cmd: List[str], not_found_msg: str) -> str:
    """Run a CLI command and return its stdout.

    Args:
        cli_name: Name of the CLI binary (e.g. "gh", "glab").
        cmd: Command arguments to pass after the binary name.
        not_found_msg: Error message to raise when the binary is not installed.

    Returns:
        Command stdout as a stripped string.

    Raises:
        PlatformError: If the command fails or the binary is not installed.
    """
    full_cmd = [cli_name] + cmd

    try:
        logger.debug("Executing: %s", " ".join(full_cmd))
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as err:
        error_msg = f"Command failed: {' '.join(full_cmd)}\n{err.stderr}"
        logger.error(error_msg)
        raise PlatformError(error_msg) from err
    except FileNotFoundError as err:
        logger.error(not_found_msg)
        raise PlatformError(not_found_msg) from err
