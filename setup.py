from setuptools import setup, find_packages

setup(
    name="OpenNMT-tf",
    version="0.1",
    install_requires=[
        "pyyaml"
    ],
    extras_require={
        "TensorFlow": ["tensorflow==1.4.0"],
        "TensorFlow (with CUDA support)": ["tensorflow-gpu==1.4.0"]
    },
    tests_require=[
        "nose2"
    ],
    test_suite="nose2.collector.collector",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "onmt = bin.main:main",
            "onmt_ark_to_records = bin.ark_to_records:main",
            "onmt_build_vocab = bin.build_vocab:main"
        ]
    }
)
