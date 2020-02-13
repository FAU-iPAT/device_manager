#! /opt/conda/bin/python3
""" General PyPI compliant setup.py configuration of the package """

# Copyright 2018 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup

version = {}
with open("device_manager/version.py") as fp:
    exec(fp.read(), version)

__author__ = ['Dominik Haspel', 'Thomas Pircher']
__version__ = version['__version__']
__copyright__ = '2020, FAU-iPAT'
__license__ = 'Apache-2.0'
__maintainer__ = 'Thomas Pircher'
__email__ = 'pi@ipat.fau.de'
__status__ = 'Development'


def get_readme() -> str:
    """
    Method to read the README.rst file

    :return: string containing README.md file
    """
    with open('readme.md') as file:
        return file.read()


# ------------------------------------------------------------------------------
#   Call setup method to define this package
# ------------------------------------------------------------------------------
setup(
    name='device_manager',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='handle device pinning for tensorflow calculations',
    long_description=get_readme(),
    url='https://github.com/FAU-iPAT/device_manager',
    license=__license__,
    keywords='???',  # todo: add keywords
    packages=['device_manager'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],  # todo: update classifiers
    python_requires='>=3.7',
    install_requires=[
        'tensorflow>=2.1',
    ],
    zip_safe=False
)
