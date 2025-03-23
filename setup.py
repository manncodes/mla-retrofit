"""
Setup script for mla-retrofit package.
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read version
with open("mla_retrofit/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mla-retrofit",
    version=version,
    description="A toolkit for retrofitting Multi-head Latent Attention (MLA) to pretrained language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MLA-Retrofit Team",
    author_email="manncodes@gmail.com",
    url="https://github.com/manncodes/mla-retrofit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": ["mla-retrofit=mla_retrofit.cli:main"],
    },
    keywords="transformers, attention, mla, gqa, llm, efficient-inference, langauge-models",
)
