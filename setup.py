#! /usr/bin/env python3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

# Arthur BERNARD

extensions=[
    Extension("script_cython", ["script_cython.pyx"],
    	include_dirs=[numpy.get_include()])
]

setup(
	cmdclass={'build_ext': build_ext},
	ext_modules=cythonize(extensions, annotate=True),
)