import os

from setuptools import setup, find_packages

base_dir = os.path.dirname(os.path.abspath(__file__))
tests_require = [
    "black==20.8b1",
    "flake8==3.8.*",
    "parameterized==0.8.1",
    "pytest-cov",
]


def get_long_description():
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, encoding="utf-8") as readme_file:
        return readme_file.read()


def get_project_version():
    version = {}
    with open(os.path.join(base_dir, "opennmt", "version.py"), encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version


version = get_project_version()
tf_version_requirement = ">=%s,<%s" % (
    version["INCLUSIVE_MIN_TF_VERSION"],
    version["EXCLUSIVE_MAX_TF_VERSION"],
)


setup(
    name="OpenNMT-tf",
    version=version["__version__"],
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
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Documentation": "https://opennmt.net/OpenNMT-tf/",
        "Forum": "https://forum.opennmt.net/",
        "Gitter": "https://gitter.im/OpenNMT/OpenNMT-tf",
        "Source": "https://github.com/OpenNMT/OpenNMT-tf/",
    },
    keywords="tensorflow opennmt nmt neural machine translation",
    python_requires=">=3.5",
    install_requires=[
        "ctranslate2>=1.18.1,<2;platform_system=='Linux' or platform_system=='Darwin'",
        "pyonmttok>=1.23.0,<2;platform_system=='Linux' or platform_system=='Darwin'",
        "pyyaml>=5.3,<5.5",
        "rouge>=1.0,<2",
        "sacrebleu>=1.5.0,<1.6",
        "tensorflow-addons>=0.12,<0.13",
    ],
    extras_require={
        "tensorflow": ["tensorflow" + tf_version_requirement],
        "tensorflow-cpu": ["tensorflow-cpu" + tf_version_requirement],
        "tests": tests_require,
    },
    tests_require=tests_require,
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
    },
)
