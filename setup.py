import setuptools
import re

# versioning ------------
VERSIONFILE="clustimage/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Setup ------------
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['datazets>=0.1.7',
                       'scatterd>=1.3.6',
                       'ismember',
                       'pypickle',
                       'pca>=2.0.4',
                       'distfit',
                       'clusteval>=2.1.5',
                       'colourmap>=1.1.16',
                       'numpy',
                       'pandas',
                       'tqdm',
                       'scikit-learn',
                       'scikit-image',
                       'matplotlib',
                       'scipy',
                       'opencv-python',
                       'imagehash'],
     python_requires='>=3',
     name='clustimage',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Python package clustimage is for unsupervised clustering of images.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://erdogant.github.io/clustimage",
	 download_url = 'https://github.com/erdogant/clustimage/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
