from __future__ import print_function

import argparse
import datetime
import re


SRC_DIR = "."
ROOT_INIT = "%s/opennmt/__init__.py" % SRC_DIR
SETUP_PY = "%s/setup.py" % SRC_DIR
DOCS_CONF = "%s/docs/conf.py" % SRC_DIR
CHANGELOG = "%s/CHANGELOG.md" % SRC_DIR


def get_current_version():
  with open(ROOT_INIT, "r") as init_file:
    for line in init_file:
      version_match = re.search('^__version__ = "(.+)"', line)
      if version_match:
        return version_match.group(1)

def replace_string_in_file(pattern, replace, path):
  with open(path, "r") as f:
    content = f.read()
  with open(path, "w") as f:
    f.write(re.sub(pattern, replace, content))

def split_version(version):
  return version.split(".")

def join_version(version):
  return ".".join(version)

def get_short_version(version):
  version = split_version(version)
  return join_version(version[0:2])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("version", help="New version to release.")
  args = parser.parse_args()

  current_version = get_current_version()
  new_version = args.version
  print("Updating version strings from %s to %s" % (current_version, new_version))

  replace_string_in_file('__version__ = "%s"' % current_version,
                         '__version__ = "%s"' % new_version,
                         ROOT_INIT)
  replace_string_in_file('version="%s"' % current_version,
                         'version="%s"' % new_version,
                         SETUP_PY)
  replace_string_in_file('release = "%s"' % current_version,
                         'release = "%s"' % new_version,
                         DOCS_CONF)
  replace_string_in_file('version = "%s"' % get_short_version(current_version),
                         'version = "%s"' % get_short_version(new_version),
                         DOCS_CONF)
  replace_string_in_file("## \[Unreleased\]",
                         "## [Unreleased]\n"
                         "\n"
                         "### New features\n"
                         "\n"
                         "### Fixes and improvements\n"
                         "\n"
                         "## [%s](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v%s) (%s)" % (
                             new_version, new_version, datetime.datetime.now().strftime("%Y-%m-%d")),
                         CHANGELOG)

  print('git tag -a v%s -m "%s release"' % (new_version, new_version))


if __name__ == "__main__":
  main()
