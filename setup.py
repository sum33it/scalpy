from setuptools import setup
from os import path
from codecs import open
from setuptools.command.install import install

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'ScalPy',
    version = '0.1.0a',
    install_requires = ['numpy', 'scipy',],

    # metadata for upload to PyPI
    author = "Sumit Kumar, Abhishek Jana and Anjan A. Sen",
    author_email = "sumit@ctp-jamia.res.in",
    description = "a package for studying dynamics of scalar fields in cosmology.",
    url = "http://github.com/sum33it/scalpy",   # project home page
    keywords = ("scalar field dynamics" + "cosmology"),
    license = "GNU GENERAL PUBLIC LICENSE v3",
    long_description = long_description,
    packages = ['scalpy',],
    classifiers = ['License :: GNU GPL v3 License',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Cosmology',
                   'Operating System :: OS Independent'
                   ]
)
