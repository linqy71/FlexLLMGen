from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="group_kvstore",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="group_kvstore",
            sources=["kvstore.cc"],
            include_dirs=["."],
            extra_compile_args=["-std=c++17", "-g", "-O0"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension 
    }
)
