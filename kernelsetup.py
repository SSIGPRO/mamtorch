from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mamtorchkernel',
      ext_modules=[
          cpp_extension.CUDAExtension(
              name='mamtorchkernel',
              sources=[# binder
                       'kernelsrc/mamtorch_bind.cpp',
                       # mam dense
                       'kernelsrc/mamdense.cpp',
                       'kernelsrc/mamdense_forward.cu',
                       'kernelsrc/mamdense_backward.cu',
                       # mam conv1d
                       'kernelsrc/mamconv1d.cpp',
                       'kernelsrc/mamconv1d_forward.cu',
                       'kernelsrc/mamconv1d_backward.cu'
                      ],
          )
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })
