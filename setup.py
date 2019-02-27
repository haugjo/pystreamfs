import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pystreamfs",
    version="0.1.0",
    author="Johannes Haug",
    author_email="johannes-christian.haug@uni-tuebingen.de",
    description="A Python package for feature selection on a simulated data stream",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haugjo/pystreamfs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
