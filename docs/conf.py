#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


# General information about the project.
project = "OpenNMT-tf"
copyright = "2018, The OpenNMT Authors"
author = "OpenNMT"
language = "en"

version = "1.25"  # The short X.Y version.
release = "1.25.0"  # The full version, including alpha/beta/rc tags.

source_suffix = ".rst"
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages"]

source_suffix = [".rst", ".md"]
source_parsers = {
   ".md": "recommonmark.parser.CommonMarkParser",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_copyright = True
html_show_sphinx = False
html_logo = "_static/logo-alpha.png"
html_favicon = "_static/favicon.png"
htmlhelp_basename = "opennmtdoc"
html_sidebars = {
    "**": [
        "relations.html",  # needs "show_related": True theme option to display
        "searchbox.html",
    ]
}

autodoc_member_order = "bysource"
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

def setup(app):
  app.add_stylesheet("custom.css")
