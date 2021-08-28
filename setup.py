import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="computational aesthetics",
    version="0.0.1",
    author="Justin Ruan",
    author_email="justin900429@gmail.com",
    description="Python package for making computational aesthetics features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Justin900429/computational_aesthetics",
    license="MIT",
    packages=["CA"],
    package_data={
        "CA": [
            "color_dict.npz"
        ]
    },
    install_requires=[
        "scikit_image==0.18.2",
        "numpy",
        "opencv_python",
        "PyWavelets",
        "matplotlib",
        "scikit_learn",
    ],
    python_requires=">=3.7"
)
