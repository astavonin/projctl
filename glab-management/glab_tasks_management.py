"""Legacy wrapper for projctl.

This script is deprecated. Use 'projctl' CLI directly.
"""

import sys
import warnings

warnings.warn(
    "glab_tasks_management.py is DEPRECATED. "
    "Use 'projctl' CLI instead.",
    DeprecationWarning,
    stacklevel=1,
)

# Print the deprecation warning to stderr explicitly so tests can detect it.
print(
    "DEPRECATION WARNING: glab_tasks_management.py is deprecated. "
    "Use 'projctl' CLI instead.",
    file=sys.stderr,
)

# Delegate to the new CLI.
from projctl.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
