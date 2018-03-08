from setuptools import setup

setup(
    name="OpenNMT-tf",
    version="0.1",
    license="MIT",
    install_requires=[
        "pyyaml"
    ],
    extras_require={
        "TensorFlow": ["tensorflow==1.6.0"],
        "TensorFlow (with CUDA support)": ["tensorflow-gpu==1.6.0"]
    },
    tests_require=[
        "nose2"
    ],
    test_suite="nose2.collector.collector",
    packages=["opennmt"]
)
