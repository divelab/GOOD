import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphOOD",
    version="0.1.0",
    author="Shurui Gui, Xiner Li",
    author_email="shurui.gui@tamu.edu",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    url="https://github.com/divelab/GOOD",
    project_urls={
        "Bug Tracker": "https://github.com/divelab/GOOD/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE Version 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"GOOD": "GOOD"},
    install_requires=[],
    python_requires=">=3.8",
)