
import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'cellcyclenet'

def read_version():
    '''Dynamically read version from authoritative _version.py file.'''
    version = {}
    with open(os.path.join(PACKAGE_NAME, '_version.py')) as f:
        exec(f.read(), version)
    return version['__version__']

setup(
    name=PACKAGE_NAME,
    version=read_version(),
    packages=find_packages(),
    include_package_data=True,  # iInclude package data as specified in MANIFEST.in
    package_data={
        PACKAGE_NAME: ['models/*.pt'],  # Include all .pt files in models/
    },
    install_requires=[
        'matplotlib==3.9.1',
        'numpy==2.0.1',
        'pandas==2.2.2',
        'scikit-image==0.24.0',
        'scikit-learn==1.5.1',
        'tifffile',
        'torch==2.3.1',
        'torchaudio==2.3.1',
        'torchvision==0.18.1',
    ],
)
