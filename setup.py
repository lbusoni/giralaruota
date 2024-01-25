#!/usr/bin/env python
import os

from setuptools import setup

NAME = 'giralaruota'
DESCRIPTION = 'who is next'
URL = ''
EMAIL = 'lorenzo.busoni@inaf.it'
AUTHOR = 'Lorenzo Busoni'
LICENSE = 'MIT'
KEYWORDS = 'Administrative, INAF, Arcetri',

here = os.path.abspath(os.path.dirname(__file__))
# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)


setup(name=NAME,
      description=DESCRIPTION,
      version=about['__version__'],
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url=URL,
      author_email=EMAIL,
      author=AUTHOR,
      license=LICENSE,
      keywords=KEYWORDS,
      packages=['giralaruota',
                ],
      install_requires=["numpy",
                        "matplotlib",
                        ],
      include_package_data=True,
      test_suite='test',
      )
