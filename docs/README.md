# Documentation

We currently use [Sphinx](http://www.sphinx-doc.org) to automatically build documentation from the code.

## Install dependencies

```bash
pip install sphinx==1.6.7
pip install sphinx_rtd_theme
pip install recommonmark
```

## Register sources for autodoc

```bash
# To run only if a Python source file is added, removed, or renamed.
rm -f docs/package/*
sphinx-apidoc -e -M -o docs/package opennmt opennmt/tests
```

## Build locally

```bash
cd docs/ && make html
```
