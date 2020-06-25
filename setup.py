import os

from setuptools import setup, find_packages

tests_require = [
    "pylint==2.4.*",
    "parameterized",
    "nose2"
]

def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()

setup(
    name="OpenNMT-tf",
    version="2.11.1",
    license="MIT",
    description="Neural machine translation and sequence learning using TensorFlow",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="OpenNMT",
    author_email="guillaume.klein@systrangroup.com",
    url="https://opennmt.net",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    project_urls={
        "Documentation": "https://opennmt.net/OpenNMT-tf/",
        "Forum": "https://forum.opennmt.net/",
        "Gitter": "https://gitter.im/OpenNMT/OpenNMT-tf",
        "Source": "https://github.com/OpenNMT/OpenNMT-tf/"
    },
    keywords="tensorflow opennmt nmt neural machine translation",
    python_requires='>=3.5',
    install_requires=[
        "ctranslate2>=1.7,<2;platform_system=='Linux'",
        "pyonmttok>=1.18.1,<2;platform_system=='Linux'",
        "pyyaml>=5.3,<5.4",
        "rouge>=1.0,<2",
        "sacrebleu>=1.4.9,<2",
        "tensorflow>=2.2,<2.3",
        "tensorflow-addons>=0.10,<0.11",
        "pyter3==0.3"
    ],
    extras_require={
        "tests": tests_require,
    },
    tests_require=tests_require,
    test_suite="nose2.collector.collector",
    packages=find_packages(exclude=["bin", "*.tests"]),
    entry_points={
        "console_scripts": [
            "onmt-ark-to-records=opennmt.bin.ark_to_records:main",
            "onmt-build-vocab=opennmt.bin.build_vocab:main",
            "onmt-detokenize-text=opennmt.bin.detokenize_text:main",
            "onmt-main=opennmt.bin.main:main",
            "onmt-merge-config=opennmt.bin.merge_config:main",
            "onmt-tokenize-text=opennmt.bin.tokenize_text:main",
        ],
    }
)
