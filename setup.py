import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "13.0")

ext_modules = [
    Pybind11Extension(
        "macldlt",
        ["macldlt.cpp"],
        cxx_std=17,
        extra_link_args=["-framework", "Accelerate"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
