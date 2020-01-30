#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


# General information about the project.
project = "OpenNMT-tf"
author = "OpenNMT"
language = "en"

version = "2.6"  # The short X.Y version.
release = "2.6.0"  # The full version, including alpha/beta/rc tags.

extensions = [
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages"]

source_suffix = [".rst", ".md"]
exclude_patterns = ["README.md"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False
html_logo = "_static/logo-alpha.png"
html_favicon = "_static/favicon.png"

autodoc_member_order = "bysource"
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

def setup(app):
  app.add_stylesheet("custom.css")
