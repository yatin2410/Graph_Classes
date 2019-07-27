import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="check_graph",
    version="1.0.0",
    author="Yatin Patel",
    author_email="yatinpatel.gt@gmail.com",
    description="This package is for graph classes recognisation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yatin2410/Graph_Classes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
