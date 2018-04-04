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
    packages=find_packages(exclude=["bin", "*.tests"]),
    entry_points={
        "console_scripts": [
            "onmt-ark-to-records=opennmt.bin.ark_to_records:main",
            "onmt-average-checkpoints=opennmt.bin.average_checkpoints:main",
            "onmt-build-vocab=opennmt.bin.build_vocab:main",
            "onmt-detokenize-text=opennmt.bin.detokenize_text:main",
            "onmt-main=opennmt.bin.main:main",
            "onmt-merge-config=opennmt.bin.merge_config:main",
            "onmt-tokenize-text=opennmt.bin.tokenize_text:main",
        ],
    }
)
