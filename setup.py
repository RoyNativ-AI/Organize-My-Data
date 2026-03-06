from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="organize-my-data",
    version="0.1.0",
    author="Roy Nativ",
    author_email="roy@officely.ai",
    description="A package to organize text data according to a specified schema",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/organize-my-data",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.41.1",
    ],
    entry_points={
        "console_scripts": [
            "organize-my-data=organize_my_data.schema_organizer:main",
        ],
    },
)
