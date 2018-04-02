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
            "opennmt.ark_to_records=bin.ark_to_records:main",
            "opennmt.average_checkpoints=bin.average_checkpoints:main",
            "opennmt.build_vocab=bin.build_vocab:main",
            "opennmt.detokenize_text=bin.detokenize_text:main",
            "opennmt.main=bin.main:main",
            "opennmt.merge_config=bin.merge_config:main",
            "opennmt.tokenize_text=bin.tokenize_text:main",
        ],
    }
)
