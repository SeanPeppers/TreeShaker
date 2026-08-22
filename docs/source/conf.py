"""Sphinx configuration for TreeShaker documentation."""

import sys
from pathlib import Path

# Add the source directory to the path so we can import treeshaker
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

project = "TreeShaker"
copyright = "2024, Sean Peppers"
author = "Sean Peppers"
release = "0.1.0"

extensions = [
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "TreeShaker — Automated Feature Selection"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": None,
    "dark_logo": None,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
    "tasklist",
]
# Auto-generate #slugified-heading anchors up to h3, so the [Foo](#foo)-style
# cross-references throughout these docs (e.g. config_reference.md's links
# to its own ### LeakageConfig/OptunaConfig/etc. sections) actually resolve
# -- without this, MyST doesn't generate heading anchors at all.
myst_heading_anchors = 3

# All docs are Markdown (no .rst files in this repo); myst_parser
# auto-registers itself for this suffix once listed in `extensions` above.
source_suffix = [".md"]
