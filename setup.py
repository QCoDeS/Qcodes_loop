from setuptools import setup
from versioningit import get_cmdclasses

# this file is kept for backwards compatibility and
# to configure versioningit. Everything else goes
# into pyproject.toml

if __name__ == "__main__":
    setup(
        cmdclass=get_cmdclasses(),
    )
