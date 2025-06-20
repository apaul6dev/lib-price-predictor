from setuptools import setup, find_packages

setup(
    name="vehicle-price-predictor",
    version="0.1.0",
    description="Librería interna para predicción de precios de vehículos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Fintech Solutions S.A.",
    author_email="dev@fintechsolutions.com",
    python_requires=">=3.8",
    packages=find_packages(include=["vehicle_price_predictor", "vehicle_price_predictor.*"]),
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "joblib",
        "pyyaml"
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
            "isort"
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
