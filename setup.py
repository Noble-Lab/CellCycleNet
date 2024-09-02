
import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'cellcyclenet'

with open("README.md", "r", encoding="utf-8") as fh:
    '''Read the contents of the README.md file.'''
    long_description = fh.read()

def read_version():
    '''Dynamically read version from authoritative _version.py file.'''
    version = {}
    with open(os.path.join(PACKAGE_NAME, '_version.py')) as f:
        exec(f.read(), version)
    return version['__version__']

setup(
    name=PACKAGE_NAME,
    url="https://github.com/Noble-Lab/CellCycleNet",
    description="Python package for predicting cell cycle stage from DAPI images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=read_version(),
    packages=find_packages(),
    include_package_data=True,  # Include package data as specified in MANIFEST.in
    package_data={
        PACKAGE_NAME: ['models/*.pt'],  # Include all .pt files in models/
    },
    install_requires=[
        'matplotlib==3.9.1',
        'numpy==2.0.1',
        'pandas==2.2.2',
        'scikit-image==0.24.0',
        'scikit-learn==1.5.1',
        'tifffile==2024.7.24',
        'torch==2.3.1',
        'torchaudio==2.3.1',
        'torchvision==0.18.1',
    ],
)
