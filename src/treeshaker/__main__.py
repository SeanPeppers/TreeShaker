"""Entry point for running TreeShaker as a module: python -m treeshaker."""

import sys

from .cli import main

sys.exit(main())
