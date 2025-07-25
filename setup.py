from setuptools import setup, find_packages
import os
import re

MODULE='darknet2any'

pkg_vars  = dict()

with open(f"{MODULE}/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), pkg_vars)

setup(name='darknet2any', packages=['darknet2any'], version=pkg_vars['__version__'])