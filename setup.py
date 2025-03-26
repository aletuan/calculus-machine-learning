from setuptools import setup, find_packages

setup(
    name="calculus_ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.8",
    author="Andy Le",
    author_email="your.email@example.com",
    description="A collection of examples demonstrating vector operations and their applications in machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aletuan/calculus-machine-learning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
) 