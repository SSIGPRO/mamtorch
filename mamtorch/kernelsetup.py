from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mamtorchkernel',
      version='0.0.3',
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
                       'kernelsrc/mamconv1d_backward.cu',
                       # mam conv2d
                       'kernelsrc/mamconv2d.cpp',
                       'kernelsrc/mamconv2d_forward.cu',
                       'kernelsrc/mamconv2d_backward.cu',
                      ],
          )
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })
