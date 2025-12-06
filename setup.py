from setuptools import setup, find_packages

setup(
    name="faircare",
    version="1.0.0",
    description="FAIR-CARE Lakehouse: Ethical AI Data Governance Pipeline",
    author="Anonymous for Review",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pyspark==3.5.0",
        "delta-spark==3.0.0",
        "pandas==2.1.3",
        "numpy==1.26.2",
        "scikit-learn==1.3.2",
        "aif360==0.6.0",
        "fairlearn==0.10.0",
        "diffprivlib==0.6.3",
        "causalnex==0.12.1",
        "dowhy==0.11.1",
        "ray==2.8.0",
        "sentence-transformers==2.2.2",
        "spacy==3.7.2",
        "presidio-analyzer==2.2.351",
        "presidio-anonymizer==2.2.351",
        "pyyaml==6.0.1",
        "requests==2.31.0",
        "streamlit==1.28.0",
        "plotly==5.17.0",
        "pytest==7.4.3",
        "pytest-cov==4.1.0",
    ],
    extras_require={
        "dev": ["black", "flake8", "mypy", "jupyter"],
    },
    entry_points={
        "console_scripts": [
            "faircare-pipeline=faircare.orchestration.pipeline:main",
        ],
    },
)
