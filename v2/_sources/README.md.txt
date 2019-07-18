# Documentation

We currently use [Sphinx](http://www.sphinx-doc.org) to automatically build documentation from the code.

## Install dependencies

```bash
pip install -r docs/requirements.txt
```

## Build locally

```bash
rm -rf docs/package
PYTHONPATH=$PWD python docs/generate-apidoc.py docs/package
PYTHONPATH=$PWD sphinx-build docs docs/build
```
