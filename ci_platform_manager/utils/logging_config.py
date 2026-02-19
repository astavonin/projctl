"""Logging configuration setup."""

import logging


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
