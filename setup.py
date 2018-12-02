from setuptools import setup

setup(
    name='SciCheat',
    url='https://github.com/andrewjkuo/scicheat',
    author='Andrew Kuo',
    packages=['scicheat'],
    install_requires=['matplotlib','numpy','pandas','scikit-learn','scipy','seaborn'],
    version='0.1dev',
    license='MIT',
    description='A package to help gain quick insights into a new dataset',
)
