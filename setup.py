import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchsummary",
    version="1.0.0",
    author="Hiram",
    author_email="bolaixv@gmail.com",
    description="everything in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inisis/brocolli",
    install_requires=["loguru",
                      "tabulate"],
    project_urls={
        "Bug Tracker": "https://github.com/inisis/brocolli/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    keywords="machine-learning, pytorch, caffe, torchfx",
)
