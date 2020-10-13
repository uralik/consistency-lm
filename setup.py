from setuptools import setup
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='self_terminating',
    version='0.0.1',
    description='self_terminating',
    packages=['self_terminating'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)