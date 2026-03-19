"""Shared utility for running glab CLI commands."""

import logging
import subprocess
from typing import List

from ..exceptions import PlatformError

logger = logging.getLogger(__name__)


def run_glab_command(cmd: List[str]) -> str:
    """Run a glab command and return its output.

    Args:
        cmd: List of command arguments to pass to glab.

    Returns:
        Command output as a string.

    Raises:
        PlatformError: If the command fails.
    """
    full_cmd = ["glab"] + cmd

    try:
        logger.debug("Executing: %s", " ".join(full_cmd))
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as err:
        error_msg = f"Command failed: {' '.join(full_cmd)}\n{err.stderr}"
        logger.error(error_msg)
        raise PlatformError(error_msg) from err
    except FileNotFoundError as err:
        error_msg = "glab command not found. Please install glab CLI."
        logger.error(error_msg)
        raise PlatformError(error_msg) from err
