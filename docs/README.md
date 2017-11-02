# Documentation

We currently use [Sphinx](http://www.sphinx-doc.org) to automatically build documentation from the code.

## Install dependencies

```bash
pip install sphinx
pip install sphinx_rtd_theme
pip install recommonmark
```

## Register sources for autodoc

```bash
# To run only if a Python source file is added, removed, or renamed.
sphinx-apidoc -e -M -o docs/package opennmt opennmt/tests
```

## Build locally

```bash
cd docs/ && make html
```
