from setuptools import setup, find_packages

setup(
    name="training",
    version="0.1.0",
    description="Simple training pipeline for census model.",
    author="Nicholas",
    packages=find_packages(exclude=('tests', 'data', 'eda', 'screenshots'))
)