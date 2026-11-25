"""Sphinx configuration for the samplepaths documentation."""

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

from docutils import nodes
from packaging.version import Version

# -- Project information ------------------------------------------------------

project = "samplepaths"
author = "Bassem Girgis"
copyright = "2018-2020 Christopher Flynn, 2026 Bassem Girgis"

try:
    _installed = Version(_dist_version("samplepaths"))
except PackageNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "samplepaths must be installed to build the documentation; run `make install` first."
    ) from exc

# The package version is derived from the git tags. A build of a release tag
# shows that version. Any other build, such as the one of the main branch, is
# labeled as the development version instead of the derived version (for
# example 0.4.1.dev3+g1a2b3c4), which reads like a release which does not
# exist.
if _installed.is_devrelease or _installed.local:
    release = "dev"
    version = "dev"
    _title_version = "(development version)"
else:
    release = str(_installed)
    version = f"{_installed.major}.{_installed.minor}"
    _title_version = release

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
language = "en"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "examples/**/README.rst",
    "examples/README.rst",
]

# Report every broken cross-reference.
nitpicky = True
nitpick_ignore = [
    # numpy documents its scalar types as attributes, not classes, so the
    # references generated from type hints cannot be resolved.
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy.object_"),
    # numpy documents NDArray as data, not as a class.
    ("py:class", "numpy.typing.NDArray"),
    # Internal base class of the processes built on Brownian motion.
    ("py:class", "samplepaths.continuous.brownian_base.BrownianBase"),
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

# The examples in docs/examples are run at build time and rendered as a
# gallery. Their figures use the plot style of the documentation.
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "gallery",
    "filename_pattern": r"plot_",
    "within_subsection_order": "FileNameSortKey",
    "reset_modules": ("matplotlib", "plot_style.reset_style"),
    "abort_on_example_error": True,
    "download_all_examples": False,
    "show_signature": False,
    "backreferences_dir": None,
}

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

# -- Readable array types ------------------------------------------------------

#: How the array types of the package are written in the API reference. The
#: type hints are expanded by autodoc to their full generic form, which is
#: hard to read.
_ARRAY_TYPES = {
    f"~numpy.ndarray[tuple[~typing.Any, ...], ~numpy.dtype[~numpy.{name}]]": (
        f"~numpy.typing.NDArray[~numpy.{name}]"
    )
    for name in ("float64", "int64", "object_")
}


def _shorten_array_types(app, domain, objtype, contentnode):  # type: ignore[no-untyped-def]
    """Rewrite the array types in the parameter and return type fields."""
    for node in contentnode.findall(nodes.Text):
        text = str(node)
        for long, short in _ARRAY_TYPES.items():
            text = text.replace(long, short)
        if text != str(node):
            node.parent.replace(node, nodes.Text(text))


def setup(app):  # type: ignore[no-untyped-def]
    """Register the hooks of this configuration."""
    # Run after autodoc has merged the type hints into the description.
    app.connect("object-description-transform", _shorten_array_types, priority=900)


# -- Options for HTML output --------------------------------------------------

# Furo follows the system color scheme and lets the reader switch between the
# light and dark themes.
html_theme = "furo"
html_title = f"samplepaths {_title_version}"
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
