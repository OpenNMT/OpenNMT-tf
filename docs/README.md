# Documentation

To build the documentation locally:

```
pip install Sphinx
pip install sphinx_rtd_theme
sphinx-apidoc -e -M -o docs opennmt opennmt/tests
cd docs/ && make html
```
