from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'rfidam.baskets_mc',
        ['rfidam/baskets_mc.pyx'],
        include_dirs=[np.get_include()],
    )
]

setup(
    name='rfidam',
    version='1.0',
    py_modules=['rfidam'],
    install_requires=[
        'Click',
        'numpy>=1.19.2',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        rfidam=rfidam.main:main
    ''',
    ext_modules=cythonize(extensions),
)
