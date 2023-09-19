from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

 
setup(name='stackarator',
       version='0.0.6',
       description='A tool for stacking interferometric data of extended sources (such as nearby galaxies) to extract weak emission lines.',
       url='https://github.com/TimothyADavis/stackarator',
       author='Timothy A. Davis',
       author_email='DavisT@cardiff.ac.uk',
       long_description=long_description,
       long_description_content_type="text/markdown",
       license='MIT',
       packages=['stackarator'],
       install_requires=[
           'numpy',
           'tqdm',
           'scipy',
           'astropy',
           'spectral_cube',
       ],
       classifiers=[
         'Development Status :: 4 - Beta',
         'License :: OSI Approved :: MIT License',
         'Programming Language :: Python :: 3',
         'Operating System :: OS Independent',
       ],
       zip_safe=True)