import io
import os
import setuptools


ROOT_DIR = os.path.dirname(__file__)


setuptools.setup(
    name="LibMultiLabel",
    version="", # Apache2.0, MIT ??
    author="LibMultiLabel Team",
    author_email="",  # libmultilabel-dev@googlegroups.com
    description=("LibMultiLabel is a simple tool with the following functionalities."
                 "end-to-end services from raw texts to final evaluation/analysis"
                 "support of common network architectures for multi-label text classification"),
    long_description=io.open(os.path.join(
        ROOT_DIR, os.path.pardir, "README.md"), "r", encoding="utf-8").read(),
    url="https://github.com/ASUS-AICS/LibMultiLabel",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "nltk",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "torch==1.8.0",
        "torchtext==0.9.0",
        "pytorch-lightning==1.3.5",
        "tqdm"
    ],
    extras_require={
        "parameter-search":[
            "bayesian-optimization",
            "optuna",
            "ray>=1.4.0",
            "ray[tune]"
        ]
    },
)
