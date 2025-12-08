FROM python:3.9-slim

# Install Java (required for Spark and ARX)
RUN apt-get update && \
    apt-get install -y openjdk-21-jdk-headless procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Install system dependencies for AIF360, Presidio, etc.
RUN apt-get update && \
    apt-get install -y build-essential git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    delta-spark==3.0.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    aif360>=0.6.0 \
    fairlearn>=0.10.0 \
    diffprivlib>=0.6.0 \
    dowhy>=0.11.0 \
    sentence-transformers>=2.2.0 \
    huggingface-hub>=0.25.0 \
    pyyaml>=6.0.0 \
    requests>=2.31.0 \
    streamlit>=1.28.0 \
    plotly>=5.17.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0

# Create app directory
WORKDIR /app

# Set python path
ENV PYTHONPATH=/app/src

CMD ["tail", "-f", "/dev/null"]
