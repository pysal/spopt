from setuptools import setup, find_packages
from distutils.command.build_py import build_py
import versioneer

package = "spopt"  # name of package

# Fetch README.md for the `long_description`
with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k, v in groups_files.items():
        with open(v, "r") as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist


def setup_package():

    _groups_files = {
        "base": "requirements.txt",  # basic requirements
        "docs": "requirements_docs.txt",  # requirements for building docs
        "dev": "requirements_dev.txt",  # requirements for development
        "tests": "requirements_tests.txt",  # requirements for testing
    }
    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop("base")
    extras_reqs = reqs

    setup(
        name=package,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass({"build_py": build_py}),
        description="Spatial Optimization in PySAL",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pysal/" + package,  # github repo
        download_url="https://pypi.org/project/" + package,
        maintainer="PySAL Developers",
        maintainer_email="xin.feng@ucr.edu, jgaboardi@gmail.com",
        keywords="spatial optimization",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        license="3-Clause BSD",
        packages=find_packages(),
        py_modules=[package],
        install_requires=install_reqs,
        extras_require=extras_reqs,
        zip_safe=False,
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    setup_package()
