# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dl-utils"
copyright = "2025-2026, Yaojie Shen"
author = "Yaojie Shen"
version = "1.0"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
]


# Make repo root importable for autodoc (useful for local builds).
import os
import sys

_DOCS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_DOCS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "yaojie-shen",
    "github_repo": "Deep-Learning-Utils",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Site icon (favicon)
html_favicon = "_static/toolbox.svg"


autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Some modules depend on optional extras (e.g. ray). We still want their
# API docs to render even when these extras are not installed in the doc build
# environment.
autodoc_mock_imports = [
    "ray",
    "ollama",
]
