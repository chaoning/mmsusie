from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


setup(
    name="mmsusie",
    version="1.0",
    author="Chao Ning",
    author_email="ningchao91@gmail.com",
    description="mmSuSiE: A Python package for mixed model SuSiE",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/chaoning/mmsusie",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.3",
        "scipy>=1.7",
        "pysnptools>=0.5",
        "tqdm>=4.60",
        "joblib>=1.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
