from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

name = "flood_fill"
ext_modules = [
    Extension(
        name,
        sources=[f"{name}.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name=name,
    ext_modules=cythonize(ext_modules),
)

setup(name='cutils', ext_modules=cythonize("cutils.pyx"))
