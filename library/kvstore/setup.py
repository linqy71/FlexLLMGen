from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="kvstore",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="kvstore",
            sources=["kvstore.cc"],
            include_dirs=["."],
            extra_compile_args=["-std=c++17", "-g", "-O0", "-D_GNU_SOURCE"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension 
    }
)
