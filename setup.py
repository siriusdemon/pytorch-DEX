import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "dex",
    version = "0.1.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="Deep EXpectation model in Pytorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/pytorch-dex",
    packages=setuptools.find_packages(),
    package_data = {
        'dex': ['pth/*.pth'],
    },
    classifiers = [
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    
)