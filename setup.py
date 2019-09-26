from setuptools import setup, find_packages

setup(
    name = "SISSOkit",
    version = "0.2.2",
    description = "Modules for cross validation, evaluation and plot of SISSO",
    long_description = "SISSOkit is a Python library for analysis of SISSO, \
        including generating cross validation files, \
        analyzing results, plotting. Data structures of SISSOkit \
        are mainly numpy array, pandas DataFrame or Series and Python \
        built-in data structure like list, so you can easily build \
        your own code based on SISSOkit.",
    license = "Apache Licence",

    url = "https://github.com/LeGenDXXX/SISSOkit",
    author = "Chuanqi Xu",
    author_email = "xcq@mail.ustc.edu.cn",

    packages = find_packages(),
    platforms = "any",
    install_requires = ['numpy','pandas','matplotlib'],
    include_package_data=True,
)
