# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:11
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : setup.py


from setuptools import setup, find_packages

setup(
    name="dl-utils",
    version="0.0.1",
    author="AcherStyx",
    author_email="acherstyx@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires='>=3.7.0',
    install_requires=[
        "numpy",
        "joblib",
        "tqdm",
        "matplotlib",
        "setuptools",
        "opencv-python",
        "torch",
        "tabulate",
        "lmdbm"
    ],
)
