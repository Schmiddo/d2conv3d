from setuptools import setup, find_packages
from torch.utils import cpp_extension

with open("README.md") as f:
    long_description = f.read()

setup(
    name="dconv_native",
    version="0.1.10",

    author="Christian Schmidt",
    author_email="christian.schmidt4@rwth-aachen.de",
    description="Cuda implementation of (modulated) deformable convolutions",
    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=["torch"],
    package_dir={"": "src"},
    packages=["dconv_native"],
    ext_modules=[
        cpp_extension.CppExtension(
            "dconv_native._ops",
            [
                "csrc/ops/dconv.cpp",
                "csrc/ops/dconv3d_gpu.cu",
                "csrc/ops/dconv1d_gpu.cu"
            ]
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
