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
        # List your package dependencies here, e.g.,
        # 'numpy',
    ],
)
