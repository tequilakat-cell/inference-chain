from setuptools import setup, find_packages

setup(
    name="inference-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "web3>=7.0.0",
        "eth-account>=0.13.0",
    ],
    python_requires=">=3.11",
)
