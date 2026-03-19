"""Legacy wrapper for ci_platform_manager.

This script is deprecated. Use 'ci-platform-manager' CLI directly.
"""

import sys
import warnings

warnings.warn(
    "glab_tasks_management.py is DEPRECATED. "
    "Use 'ci-platform-manager' CLI instead.",
    DeprecationWarning,
    stacklevel=1,
)

# Print the deprecation warning to stderr explicitly so tests can detect it.
print(
    "DEPRECATION WARNING: glab_tasks_management.py is deprecated. "
    "Use 'ci-platform-manager' CLI instead.",
    file=sys.stderr,
)

# Delegate to the new CLI.
from ci_platform_manager.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
