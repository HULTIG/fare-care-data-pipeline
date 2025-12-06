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
# Note: arx4py might need specific installation instructions or a wheel if not on PyPI. 
# Assuming standard pip installable packages for now or using alternatives if needed.
# For this blueprint, we will install standard packages.
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    delta-spark==3.0.0 \
    pandas==1.5.3 \
    numpy==1.23.5 \
    scikit-learn==1.3.2 \
    aif360==0.6.0 \
    fairlearn==0.8.0 \
    diffprivlib==0.6.3 \
    causalnex==0.12.1 \
    dowhy==0.11.1 \
    ray==2.8.0 \
    sentence-transformers==2.2.2 \
    spacy==3.7.2 \
    presidio-analyzer==2.2.351 \
    presidio-anonymizer==2.2.351 \
    pyyaml==6.0.1 \
    requests==2.31.0

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create app directory
WORKDIR /app

# Copy source code and configs (will be mounted in dev, but good for prod)
# COPY src /app/src
# COPY configs /app/configs
# COPY scripts /app/scripts

# Set python path
ENV PYTHONPATH=/app/src

CMD ["tail", "-f", "/dev/null"]
