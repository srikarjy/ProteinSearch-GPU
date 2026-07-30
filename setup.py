from setuptools import setup, find_packages
import os
import torch

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    cuda_sources = [
        'cuda/normalize.cu',
        'cuda/cosine.cu',
        'cuda/topk.cu',
        'cuda/gather.cu',
        'cuda/smith_waterman.cu',
    ]
    
    ext_modules = [
        CUDAExtension(
            name='protein_search_gpu._C',
            sources=['bindings/torch_extension.cpp'] + cuda_sources,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--expt-extended-lambda',
                    '-use_fast_math',
                ],
            },
        ),
    ]
    
    cmdclass = {'build_ext': BuildExtension}
else:
    ext_modules = []
    cmdclass = {}

setup(
    name='protein_search_gpu',
    version='0.1.0',
    description='CUDA-Accelerated Semantic Protein Retrieval',
    author='ProteinSearch-GPU',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'torch>=2.0',
        'numpy',
        'transformers>=4.30',
        'accelerate',
        'tqdm',
        'h5py',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)