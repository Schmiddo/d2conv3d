from setuptools import setup, find_packages
from torch.utils import cpp_extension

with open("README.md") as f:
    long_description = f.read()

setup(
    name="fast-depthwise-conv3d",
    version="0.0.2",

    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=["torch"],
    package_dir={"": "src"},
    packages=["fast_depthwise_conv3d"],
    ext_modules=[
        cpp_extension.CppExtension(
            "fast_depthwise_conv3d._ops",
            [
                "csrc/python_bindings.cpp",
                "csrc/grouped_conv3d.cu"
            ]
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
