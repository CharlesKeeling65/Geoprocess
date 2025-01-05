from setuptools import setup, find_packages

setup(
    name="geoprocess",
    version="1.0.0",
    description="A package for geospatial processing",
    author="Charles Keeling",
    author_email="charleskeeling65@163.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "pathlib",
        "rasterio",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
