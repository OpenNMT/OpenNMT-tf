from setuptools import setup, find_packages

tests_require = [
    "nose2"
]

setup(
    name="OpenNMT-tf",
    version="1.8.0",
    license="MIT",
    description="Neural machine translation and sequence learning using TensorFlow",
    author="OpenNMT",
    author_email="guillaume.klein@opennmt.net",
    url="http://opennmt.net",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    project_urls={
        "Documentation": "http://opennmt.net/OpenNMT-tf/",
        "Forum": "http://forum.opennmt.net/",
        "Gitter": "https://gitter.im/OpenNMT/OpenNMT-tf",
        "Source": "https://github.com/OpenNMT/OpenNMT-tf/"
    },
    keywords="tensorflow opennmt nmt neural machine translation",
    install_requires=[
        "pyonmttok>=1.5.0,<2;platform_system=='Linux'",
        "pyyaml",
        "rouge==0.3.1"
    ],
    extras_require={
        "tests": tests_require,
        "tensorflow": ["tensorflow>=1.4.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.4.0"]
    },
    tests_require=tests_require,
    test_suite="nose2.collector.collector",
    packages=find_packages(exclude=["bin", "*.tests"]),
    package_data={
        "opennmt": [
            "../third_party/multi-bleu.perl",
            "../third_party/multi-bleu-detok.perl"]
    },
    entry_points={
        "console_scripts": [
            "onmt-ark-to-records=opennmt.bin.ark_to_records:main",
            "onmt-average-checkpoints=opennmt.bin.average_checkpoints:main",
            "onmt-build-vocab=opennmt.bin.build_vocab:main",
            "onmt-detokenize-text=opennmt.bin.detokenize_text:main",
            "onmt-main=opennmt.bin.main:main",
            "onmt-merge-config=opennmt.bin.merge_config:main",
            "onmt-tokenize-text=opennmt.bin.tokenize_text:main",
            "onmt-update-vocab=opennmt.bin.update_vocab:main",
        ],
    }
)
