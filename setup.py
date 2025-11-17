import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='identity_exchange_profiler',
    version='0.1.1',
    packages=['identity_exchange_profiler'],
    ext_modules=cythonize(["identity_exchange_profiler/process_traces_cython.pyx"]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
