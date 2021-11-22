import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pcurvepy", # Replace with your own username
    version="1.0",
    author="Jacob Moss",
    description="Principal curves implementation in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mossjacob/pcurvepy",
    packages=["pcurve"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
