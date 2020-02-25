
from setuptools import setup

setup(name='timeseries',
      version='0.1',
      description='time series objects',
      url='http://github.com/goiosunw/timeseries',
      author='Andre Goios',
      author_email='a.almeida@unsw.edu.au',
      license='GPL v3',
      packages=['timeseries'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
      zip_safe=False)
