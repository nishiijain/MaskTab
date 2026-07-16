from setuptools import setup, find_packages

setup(
    name="masktab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==2.5.1",
        "pandas==3.0.3",
        "pytest==9.1.1",
        "scikit-learn==1.9.0",
        "torch==2.13.0",
        "tqdm==4.67.1",
        "click>=8.0.0",
        "requests>=2.31.0"
    ],
    entry_points={
        "console_scripts": [
            "masktab=masktab.cli:cli",
        ],
    },
)
