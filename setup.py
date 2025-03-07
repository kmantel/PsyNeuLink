import versioneer

# Always prefer setuptools over distutils
from setuptools import setup

setup(
    name='psyneulink',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
