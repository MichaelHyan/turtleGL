from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="TurtleGL-3d",
    version="1.2.2",
    author="Han Yan",
    author_email="3367461801@qq.com",
    description="A 3D grafics library based on turtle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelHyan/turtleGL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.4",
        "opencv-python>=4.12.0"
    ],
)