import setuptools

with open("README.md", "r", encoding="utf_8") as file:
    package_description = file.read()

setuptools.setup(
    name="Housing-ApurvTA",
    version="0.1",
    keywords=["housing data", "sample package"],
    description="In this package a model is created and implemented on housing data",
    long_description=package_description,
    long_description_content_type="text/markdown",
    author="Apurv Master",
    author_email="apurv.master@tigeranalytics.com",
    url="https://github.com/Apurv-TA/mle-training",
    project_urls={
        "Bug Tracker": "https://github.com/Apurv-TA/mle-training/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)
