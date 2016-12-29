# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from mdsvm import __license__, __author__, __version__

setup(
    name="tfmodel",
    description="",
    version=__version__,
    license=__license__,
    author=__author__,
    author_email="shuhei.fujiwara@gmail.com",
    url="https://github.com/sfujiwara/tfmodel",
    packages=find_packages(),
    # install_requires=["tensorflow"],
)
