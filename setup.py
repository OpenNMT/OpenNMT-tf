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
    packages=find_packages(exclude=["*.tests"]),
    entry_points={
        "console_scripts": [
            "bin.ark_to_records=opennmt.bin.ark_to_records:main",
            "bin.average_checkpoints=opennmt.bin.average_checkpoints:main",
            "bin.build_vocab=opennmt.bin.build_vocab:main",
            "bin.detokenize_text=opennmt.bin.detokenize_text:main",
            "bin.main=opennmt.bin.main:main",
            "bin.merge_config=opennmt.bin.merge_config:main",
            "bin.tokenize_text=opennmt.bin.tokenize_text:main",
        ],
    }
)
