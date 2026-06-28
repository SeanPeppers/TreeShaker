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

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
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

# Support both .rst and .md files
source_suffix = {
    ".rst": None,
    ".md": None,
}
