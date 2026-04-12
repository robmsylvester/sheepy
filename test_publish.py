"""Pre-publish check: ensure package version is newer than the latest git tag."""
import re
import subprocess
from pathlib import Path


def test_bump_version_number():
    init_file = Path(__file__).parent / "sheepy" / "__init__.py"
    version_str = re.search(r'__version__ = "([^"]+)"', init_file.read_text()).group(1)

    result = subprocess.run(
        ["git", "describe", "--abbrev=0", "--tags"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return  # No tags yet

    latest = result.stdout.strip().lstrip("v")
    latest_parts = list(map(int, latest.split(".")))
    current_parts = list(map(int, version_str.split(".")))
    assert current_parts > latest_parts, (
        f"Version {version_str} is not newer than latest tag {latest}"
    )
