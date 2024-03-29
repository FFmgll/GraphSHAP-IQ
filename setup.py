import codecs
import os

import setuptools

NAME = "shapiq"
DESCRIPTION = "SHAPley Interaction Quantification (SHAP-IQ) for Explainable AI"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://github.com/mmschlk/shapiq"
EMAIL = "maximilian.muschalik@ifi.lmu.de"
AUTHOR = "Maximilian Muschalik Fabian Fumagalli"
REQUIRES_PYTHON = ">=3.9.0"

work_directory = os.path.abspath(os.path.dirname(__file__))


# https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    with codecs.open(os.path.join(work_directory, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delimiter = '"' if '"' in line else "'"
            return line.split(delimiter)[1]


with open(os.path.join(work_directory, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

base_packages = ["numpy", "scipy", "pandas", "tqdm"]

plotting_packages = ["matplotlib", "colour", "networkx"]

doc_packages = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx_rtd_theme",
    "sphinx_toolbox",
    # "myst_nb",
    "nbsphinx",  # for rendering jupyter notebooks
    "pandoc",  # for rendering jupyter notebooks
    "furo",  # theme of the docs
    "sphinx-copybutton",  # easier copy-pasting of code snippets from docs
    "myst-parser",  # parse md and rst files
]

dev_packages = [
    "build",
    "black",
    "pytest",
    "coverage",
    "scikit-learn",
]

setuptools.setup(
    name=NAME,
    version=get_version("shapiq/__init__.py"),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls={
        "Tracker": "https://github.com/mmschlk/shapiq/issues?q=is%3Aissue+label%3Abug",
        "Source": "https://github.com/mmschlk/shapiq",
    },
    packages=setuptools.find_packages(include=("shapiq", "shapiq.*")),
    install_requires=base_packages + plotting_packages,
    extras_require={
        "docs": base_packages + plotting_packages + doc_packages,
        "dev": base_packages + plotting_packages + doc_packages + dev_packages,
    },
    include_package_data=True,
    license="MIT",
    license_files=("LICENSE",),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "python",
        "machine learning",
        "shap",
        "xai",
        "interaction",
        "shapley interactions",
        "shapley values",
        "feature interaction",
    ],
    zip_safe=True,
)
