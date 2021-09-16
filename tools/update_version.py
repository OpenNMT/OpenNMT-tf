import argparse
import datetime
import re

SRC_DIR = "."
VERSION_FILE = "%s/opennmt/version.py" % SRC_DIR
CHANGELOG = "%s/CHANGELOG.md" % SRC_DIR


def get_current_version():
    with open(VERSION_FILE, "r") as version_file:
        for line in version_file:
            version_match = re.search('^__version__ = "(.+)"', line)
            if version_match:
                return version_match.group(1)


def replace_string_in_file(pattern, replace, path):
    with open(path, "r") as f:
        content = f.read()
    with open(path, "w") as f:
        f.write(re.sub(pattern, replace, content))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="New version to release.")
    args = parser.parse_args()

    current_version = get_current_version()
    new_version = args.version
    print("Updating version strings from %s to %s" % (current_version, new_version))

    replace_string_in_file(
        '__version__ = "%s"' % current_version,
        '__version__ = "%s"' % new_version,
        VERSION_FILE,
    )
    replace_string_in_file(
        r"## \[Unreleased\]",
        "## [Unreleased]\n"
        "\n"
        "### New features\n"
        "\n"
        "### Fixes and improvements\n"
        "\n"
        "## [%s](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v%s) (%s)"
        % (new_version, new_version, datetime.datetime.now().strftime("%Y-%m-%d")),
        CHANGELOG,
    )

    print('git tag -a v%s -m "%s release"' % (new_version, new_version))


if __name__ == "__main__":
    main()
