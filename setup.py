import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RL_Projects_Jakkes", # Replace with your own username
    version="0.0.1",
    author="Jakob Stigenberg",
    author_email="jakob@jakke.se",
    description="RL Projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakkes/RL_Projects",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)