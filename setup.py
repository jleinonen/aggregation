#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

long_description = """A Python code for generating 3D models of aggregate
and rimed snowflakes.
"""

setup(name='aggregation',
      description='Aggregate snowflake generator',
      author='Jussi Leinonen',
      author_email='jsleinonen@gmail.com',
      packages=['aggregation'],
      package_data = {
            'aggregation': ['dendrite_grid.dat'],            
      },
      long_description = long_description,
     )
