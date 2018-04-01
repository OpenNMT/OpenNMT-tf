from setuptools import setup, find_packages

setup(
    name="OpenNMT-tf",
    version="0.1",
    license="MIT",
    install_requires=[
        "pyyaml"
    ],
    extras_require={
        "tensorflow": ["tensorflow==1.6.0"],
        "tensorflow_gpu": ["tensorflow-gpu==1.6.0"]
    },
    tests_require=[
        "nose2"
    ],
    test_suite="nose2.collector.collector",
    packages=find_packages(exclude=["bin", "*.tests"])
)
