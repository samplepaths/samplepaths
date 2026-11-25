"""Sphinx configuration for the samplepaths documentation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

# -- Project information ------------------------------------------------------

project = "samplepaths"
author = "Bassem Girgis"
copyright = "2018-2020 Christopher Flynn, 2026 Bassem Girgis"

try:
    release = _dist_version("samplepaths")
except PackageNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "samplepaths must be installed to build the documentation; "
        "run `uv sync --dev` (or `pip install .`) first."
    ) from exc

version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Report every broken cross-reference.
nitpicky = True
nitpick_ignore = [
    # numpy documents its scalar types as attributes, not classes, so the
    # references generated from type hints cannot be resolved.
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy.object_"),
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Generate the API reference pages from the autosummary directives.
autosummary_generate = True
# Document the public names a package exports through ``__all__``.
autosummary_ignore_module_all = False
templates_path = ["_templates"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- Options for HTML output --------------------------------------------------

# Furo follows the system color scheme and lets the reader switch between the
# light and dark themes.
html_theme = "furo"
html_title = f"samplepaths {release}"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    # Brand colors matching the palette of the figures.
    "light_css_variables": {
        "color-brand-primary": "#4f46e5",
        "color-brand-content": "#4f46e5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a5b4fc",
        "color-brand-content": "#a5b4fc",
    },
    "source_repository": "https://github.com/samplepaths/samplepaths/",
    "source_branch": "master",
    "source_directory": "docs/",
}
pygments_style = "default"
pygments_dark_style = "monokai"
htmlhelp_basename = "samplepathsdoc"

# -- Options for other builders -----------------------------------------------

latex_documents = [
    (
        master_doc,
        "samplepaths.tex",
        "samplepaths Documentation",
        author,
        "manual",
    )
]

man_pages = [(master_doc, "samplepaths", "samplepaths Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "samplepaths",
        "samplepaths Documentation",
        author,
        "samplepaths",
        "Generate realizations of stochastic processes.",
        "Miscellaneous",
    )
]
