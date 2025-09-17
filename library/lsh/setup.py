from torch.utils.cpp_extension import CppExtension
import setuptools

ext_modules = [
    CppExtension(
        "lsh",
        sources=[
            "lsh.cc"
        ],
        extra_compile_args=["-mavx512f", "-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0","-std=c++17", '-O3', "-g"],
        extra_link_args=['-fopenmp', '-O3', '-g'],
    ),
]

setuptools.setup(name="lsh", version="0.1.0", ext_modules=ext_modules)