from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RAG for Department of Data Science",
    version="0.1",
    author="Ekaansh",
    packages=find_packages(),
    install_requires = requirements,
)