import json
import urllib.request
from setuptools import setup


def get_version():
    """Get the version number."""
    with urllib.request.urlopen(
        "https://api.github.com/repos/silvxlabs/quicfire-tools/releases/latest"
    ) as response:
        data = json.loads(response.read().decode("utf-8"))
    version = data["tag_name"]
    return version[1:]  # Remove the leading "v" from the version number


setup(version=get_version())
