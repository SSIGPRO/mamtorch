from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mamtorchkernel',
      ext_modules=[
          cpp_extension.CUDAExtension(
              name='mamtorchkernel',
              sources=['kernelsrc/mamtorch_bind.cpp',
                       'kernelsrc/mamdense.cpp',
                       'kernelsrc/mamdense_cuda.cu',
                      ],
          )
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })
