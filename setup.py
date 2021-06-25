#!/usr/bin/env python3

from imp import load_source
from os import path
from setuptools import setup

PKGINFO = load_source('pkginfo.version', 'src/version.py')
__version__ = PKGINFO.__version__
__packagename__ = PKGINFO.__packagename__

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(name=__packagename__,
      version=__version__,
      description='An implementation of retypd in Python3',
      author='GrammaTech, Inc.',
      package_dir={__packagename__: 'src'},
      packages=[__packagename__],
      install_requires=install_requires,
      )
