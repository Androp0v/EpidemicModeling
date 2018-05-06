# from distutils.core import setup
# from Cython.Build import cythonize

# setup(
#     ext_modules = cythonize("UpdateCoefficients.pyx")
# )

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("UpdateCoefficients",
              ["UpdateCoefficients.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "UpdateCoefficients",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)