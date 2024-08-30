from setuptools import setup, find_packages

setup(
    name="cellcyclenet",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Include package data as specified in MANIFEST.in
    package_data={
        'cellcyclenet': ['models/*.pt'],  # Include all .pt files in models/
    },
    install_requires=[
        # List your package dependencies here, e.g.,
        # 'numpy',
    ],
)
