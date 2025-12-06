# Horus Legal ETL Pipeline: Technical Architecture & Methodology

> **Document Status**: Active
> **Author**: HULTIG
> **Version**: 1.0
> **Date**: 2025-11-28

---

## 1. Executive Summary

The **Horus Legal ETL Pipeline** is a state-of-the-art, distributed data processing system designed to ingest, transform, and semantically enrich legal documents at scale. It leverages the **Medallion Architecture** pattern within a **Lakehouse** paradigm, combining the reliability of data warehouses with the flexibility of data lakes.

This system is engineered to solve the challenge of unstructured legal data analysis by integrating **Large Language Models (LLMs)** for cognitive processing and **Vector Databases** for semantic retrieval, all orchestrated via **Apache Spark** and **Docker**.

---

## 2. Architectural Philosophy

Our design philosophy centers on four pillars:

1.  **Scalability**: Decoupled storage (Delta Lake) and compute (Spark) allow independent scaling.
2.  **Reliability**: ACID transactions via Delta Lake ensure data integrity during concurrent writes.
3.  **Modularity**: Microservices architecture (FastAPI, Celery, Qdrant) enables independent deployment and maintenance.
## 5. The ETL Methodology: Medallion Architecture

We implement a strict **Bronze-Silver-Gold** pipeline to ensure data quality and lineage.

### 5.1 Bronze Layer (Raw Ingestion)
- **Objective**: Capture data in its native format without loss.
- **Implementation**: `create_sample_bronze.py`
- **Characteristics**:
    - **Schema-on-Write**: Minimal validation.
    - **Immutability**: Data is appended, rarely overwritten.
    - **Format**: Parquet (via Delta) containing raw HTML/Text.

### 5.2 Silver Layer (Refinement)
- **Objective**: Clean, deduplicate, and validate data.
- **Implementation**: `silver_transformation.py`
- **Techniques**:
    - **HTML Parsing**: `BeautifulSoup` removes markup artifacts.
    - **Validation**: Rules-based filtering (e.g., `word_count > 10`).
    - **Deduplication**: Enforced uniqueness on `document_id`.
    - **Schema Enforcement**: Strict typing applied here.

### 5.3 Gold Layer (Business Aggregation & Enrichment)
- **Objective**: Apply business logic and AI enrichment.
- **Implementation**: `gold_enhancement.py`
- **Techniques**:
    - **LLM Integration**: Calls to Ollama to generate:
        - *Executive Summaries*: Compression of legal text.
        - *NER (Named Entity Recognition)*: Extraction of Parties, Dates, Jurisdictions.
        - *Topic Modeling*: Classification into legal domains (e.g., "Commercial Law").
    - **Structured Output**: LLM outputs are parsed into strict JSON schemas for downstream consumption.

---

## 6. Semantic Search & Vectorization

The key part from the pipeline is the semantic search capability, bridging the gap between keyword search and intent understanding.

### 6.1 Embedding Generation
- **Process**: `add_embeddings.py`
- **Model**: `nlpaueb/legal-bert-base-uncased` (or mock equivalent for dev).
- **Dimensionality**: 768 dimensions.
- **Strategy**: We embed the *Executive Summary* rather than the full text to capture the core semantic meaning within the model's context window.

### 6.2 Vector Indexing
- **Process**: `qdrant_indexing.py`
- **Storage**: Qdrant Collection `legal_documents`.
- **Payload**: We store the Gold layer metadata (Entities, Topics) alongside vectors to enable **Hybrid Search** (Semantic Vector Search + Metadata Filtering).

---

## 7. Key Engineering Fundamentals

### 7.1 Idempotency
All Spark jobs are designed to be idempotent. Running the same job twice results in the same state, preventing data duplication. This is achieved via Delta Lake's `merge` (upsert) operations.

### 7.2 Fault Tolerance
- **Spark**: RDD lineage allows recovery from worker node failures.
- **Celery**: `acks_late` configuration ensures tasks are not lost if a worker crashes during processing.

### 7.3 Asynchronous Processing
User uploads are handled asynchronously. The API accepts the file, returns a `task_id`, and offloads processing to Celery. This prevents HTTP timeouts during long-running OCR or ingestion tasks.

---

## 8. Future Roadmap

To evolve this into an enterprise-grade platform, the following steps are recommended:

1.  **Orchestration**: Replace manual `spark-submit` with **Apache Airflow** or **Dagster** for DAG management.
2.  **Data Quality**: Integrate **Great Expectations** for automated data testing between layers.
3.  **CI/CD**: Implement GitHub Actions for automated testing of Spark jobs and API endpoints.
4.  **Security**: Implement Role-Based Access Control (RBAC) on the API and encryption at rest for the Data Lake.

---

