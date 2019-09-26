from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='scicheat',
    url='https://github.com/andrewjkuo/scicheat',
    author='Andrew Kuo',
    packages=['scicheat'],
    install_requires=['matplotlib','numpy','pandas','scikit-learn','seaborn'],
    version='0.3',
    license='MIT',
    description='A package to help gain quick insights into a new dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
