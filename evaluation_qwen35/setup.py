from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
version_ns: dict[str, str] = {}
exec((this_directory / "flashrag" / "version.py").read_text(), version_ns)
__version__ = version_ns["__version__"]


def _read_requirements(path: str = "requirements.txt") -> list[str]:
    req_path = this_directory / path
    lines = req_path.read_text().splitlines()
    reqs: list[str] = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(line)
    return reqs


setup(
    name="evaluation-qwen35",
    version=__version__,
    packages=find_packages(include=["flashrag", "flashrag.*"]),
    install_requires=_read_requirements("requirements.txt"),
    package_data={"": ["**/*.yaml"]},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
