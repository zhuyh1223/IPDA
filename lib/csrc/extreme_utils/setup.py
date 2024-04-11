from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob

cxx_args = []

nvcc_args = [
'-D__CUDA_NO_HALF_OPERATORS__' # <-- Just add this line
]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    main_file = glob.glob(os.path.join(this_dir, '*.cpp'))
    source_cuda = glob.glob(os.path.join(this_dir, 'src', '*.cu'))
    sources = main_file + source_cuda
    include_dirs = [this_dir]
    ext_modules = [
        CUDAExtension(
            name='_ext',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
        )
    ]
    return ext_modules


setup(
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension}
)
