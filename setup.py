from setuptools import setup, find_packages

setup(
    name="rag_experiments",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Requirements will be read from requirements.txt
    ],
    description="A package for experimenting with Retrieval Augmented Generation systems",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)