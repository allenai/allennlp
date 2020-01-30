from setuptools import find_namespace_packages, setup


setup(
    name="c",
    version="0.1.0",
    description="Test C",
    url="https://test.com",
    author="AI Author",
    author_email="test@test.com",
    license="Apache 2.0",
    packages=["c"] + find_namespace_packages(include=["allennlp_plugins.*"]),
)
