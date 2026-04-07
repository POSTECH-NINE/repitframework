from setuptools import setup, find_packages

setup(
    name="repitframework",  # Name of the package
    version="1.0",  # Version of the package
    author="Shilaj Baral",
    author_email="shilaj@postech.ac.kr",
    description="Automation framework for ML-CFD cross-computation.",
    long_description=open("README.md").read(),  # Can read from README
    long_description_content_type="text/markdown",  # Specify format
    url="https://github.com/POSTECH-NINE/repitframework",  # Project URL
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        "numpy",   # List of dependencies
        "pandas",
        "Ofpp",
        "torch",
        "imageio",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.12',  # Minimum Python version
)
