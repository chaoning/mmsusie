from setuptools import setup, find_packages

setup(
    name="mmsusie",
    version="0.1.0",
    author="Chao Ning",
    author_email="ningchao91@gmail.com",
    description="mmSuSiE: A Python package for mixed model SuSiE",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mmsusie",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "logging"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
