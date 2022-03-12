from setuptools import setup, find_packages

setup(
   name='graph_sim_match',
   version='1.0.0',
   author='Sergio A. Serrano',
   author_email='sserrano@inaoep.mx',
   description='Implementation of two graph similarity/matching methods.',
   packages=find_packages(where='.'),
   python_requires='==3.7.11',
   install_requires=[
       'numpy == 1.21.4',
       'pandas == 1.3.4',
       'scipy == 1.7.3',
       'ortools == 9.1.9490'
   ]
)
