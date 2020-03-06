from setuptools import find_namespace_packages, setup


setup(
    name="allennlp-server",
    version="0.1.0",
    description="AllenNLP Server",
    url="https://test.com",
    author="AI Author",
    author_email="test@test.com",
    license="Apache 2.0",
    packages=["allennlp_server"] + find_namespace_packages(include=["allennlp_plugins.*"]),
)
