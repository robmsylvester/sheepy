import os
import re
import subprocess


def test_bump_version_number():
    PACKAGE_NAME = "sheepy"
    HERE = os.path.abspath(os.path.dirname(__file__))
    PACKAGE_INIT_FILE = os.path.join(HERE, PACKAGE_NAME, "__init__.py")
    with open(PACKAGE_INIT_FILE, encoding="utf-8") as fp:
        VERSION = re.search('__version__ = "([^"]+)"', fp.read()).group(1)
    latest_version = subprocess.run(
        ["git", "describe", "master", "--abbrev=0", "--tags"], text=True, capture_output=True
    ).stdout
    latest_version = list(map(int, latest_version.split(".")))
    to_publish_version = list(map(int, VERSION.split(".")))
    assert to_publish_version > latest_version
