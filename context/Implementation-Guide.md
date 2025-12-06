# FAIR-CARE Lakehouse: Detailed Implementation Guide for Researchers and Data Engineers

**For: ICSA 2026 Artifact Submission**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Algorithm Implementation Details](#algorithm-implementation-details)
4. [Data Pipeline Walkthrough](#data-pipeline-walkthrough)
5. [Experiment Reproduction](#experiment-reproduction)
6. [Evaluation Metrics Explained](#evaluation-metrics-explained)
7. [Troubleshooting and FAQs](#troubleshooting-and-faqs)

---

## Quick Start

### Prerequisites

- Python 3.9+
- Apache Spark 3.3+
- Docker & Docker Compose (recommended)
- 16GB RAM minimum (8GB minimum for demo)
- macOS, Linux, or Windows (WSL2)

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/[org]/fair-care-lakehouse.git
cd fair-care-lakehouse

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Start full stack with Docker
docker-compose up -d  # Spark, Delta Lake, Postgres, Streamlit

# 5. Download datasets
python scripts/download_datasets.py --datasets compas,adult,german_credit
```

### First Run (10 minutes)

```bash
# Run pipeline on COMPAS dataset with default configuration
python -m fair_care.orchestration.pipeline \
  --dataset compas \
  --config configs/default.yaml \
  --output results/run_001 \
  --verbose

# Expected output:
# ✓ Bronze layer: 847 records ingested, 12 PII fields detected
# ✓ Silver layer: k-anonymity (k=5) applied, utility=0.94
# ✓ Gold layer: fairness metrics computed, DPD=0.12, EOD=0.16
# ✓ FAIR-CARE Score: 0.80
```

### View Results

```bash
# Launch interactive dashboard
streamlit run src/dashboard/app.py

# Open browser: http://localhost:8501
# Explore: metrics, causal graphs, fairness visualizations
```

---

## Architecture Deep Dive

### Layer 1: Bronze – Raw Ingestion with Governance

**Responsibility:** Ingest raw data, detect sensitive information, establish baseline fairness metrics.

**Key Components:**

```python
from fair_care.bronze import DataIngestion, PIIDetection, BiasAudit

# Initialize ingestion
ingestion = DataIngestion(
    source="path/to/compas.csv",
    source_system="Broward County COMPAS",
    steward_contact="research@example.org"
)

# Ingest data
df_bronze = ingestion.ingest()
# Returns: Spark DataFrame + metadata

# Detect PII
pii_detector = PIIDetection(
    confidence_threshold=0.95,
    techniques=["regex", "nlp", "statistical"]
)
pii_report = pii_detector.detect(df_bronze)
# Returns: {column: [pii_types], confidence: [scores]}

# Audit for bias
bias_auditor = BiasAudit(
    protected_attributes=["race", "gender"],
    quasi_identifiers=["age", "postcode"]
)
bias_baseline = bias_auditor.compute_baseline(df_bronze)
# Returns: {
#   "race": {"African American": 0.52, "Caucasian": 0.48},
#   "gender": {"Male": 0.55, "Female": 0.45},
#   "imbalance_ratio": 1.08
# }
```

**Bronze Layer Metrics:**

```python
from fair_care.metrics import BronzeMetrics

metrics_b = BronzeMetrics(
    ingestion_metadata=ingestion.metadata,
    pii_report=pii_report,
    bias_baseline=bias_baseline
)

# Calculate S_B (Bronze Score)
s_b = metrics_b.compute_score()
print(f"Provenance Score: {metrics_b.provenance_score}")     # 0.98
print(f"PII Detection Rate: {metrics_b.pii_detection_rate}")  # 0.96
print(f"Quality Baseline: {metrics_b.quality_baseline}")      # 0.92
print(f"Bronze Score (S_B): {s_b}")                           # 0.95
```

**Output Storage:**

```python
# Save to Delta Lake (versioned, ACID-compliant)
df_bronze.write.format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .save("s3://lakehouse/bronze/compas")

# Store metadata
metadata_df = spark.createDataFrame([{
    "source_id": "compas_2025_01",
    "ingestion_timestamp": "2025-01-15T10:30:00Z",
    "record_count": 6214,
    "pii_detected": True,
    "pii_fields": ["defendant_id", "name", "address"],
    "demographic_baseline": bias_baseline
}])
metadata_df.write.format("delta").mode("append").save(...)
```

---

### Layer 2: Silver – Anonymization & Causal Analysis

**Responsibility:** Apply privacy techniques, validate causal assumptions, assess utility trade-offs.

#### 2a. Anonymization Engine

```python
from fair_care.silver.anonymization import AnonymizationEngine

# Configure anonymization
config = {
    "technique": "differential_privacy",  # or "k_anonymity", "l_diversity", "t_closeness"
    "epsilon": 0.1,                       # DP parameter
    "delta": 1e-6,
    "quasi_identifiers": ["age", "postcode", "gender"],
    "sensitive_attributes": ["prior_arrests", "recidivism_risk"]
}

# Initialize engine
anonymizer = AnonymizationEngine(config=config)

# Run anonymization
df_anon, anon_report = anonymizer.anonymize(df_bronze)

# Examine report
print(f"Anonymization Report:")
print(f"  Technique: {anon_report.technique}")
print(f"  K-value achieved: {anon_report.k_achieved}")           # 5
print(f"  Records suppressed: {anon_report.suppressed_count}")  # 23 (0.37%)
print(f"  Information loss (NCP): {anon_report.info_loss}")     # 0.078
print(f"  Privacy risk (re-id): {anon_report.privacy_risk}")    # 0.03
```

**Under the hood (ARX Integration):**

```python
# fair_care/silver/anonymization.py (excerpt)
import arx4py

class AnonymizationEngine:
    def anonymize(self, df, config):
        # Convert Spark DF to Pandas for ARX
        data_pandas = df.select(...).toPandas()
        
        # Configure ARX
        arx_config = arx4py.ARXConfiguration()
        arx_config.addPrivacyModel(
            privacy_model=arx4py.KAnonymity(config["k_value"])
        )
        
        # Specify quasi-identifiers
        for qi in config["quasi_identifiers"]:
            arx_config.addAttribute(qi, attribute_type="quasi-identifying")
        
        # Apply anonymization
        result = arx4py.anonymize(data_pandas, arx_config)
        
        # Convert back to Spark
        df_anon = spark.createDataFrame(result.output)
        
        return df_anon, result.metadata
```

#### 2b. Causal Graph Analysis

```python
from fair_care.silver.causal_analysis import CausalAnalyzer

# Define causal assumptions (domain expert input)
causal_graph = {
    "nodes": ["race", "ses", "prior_arrests", "recidivism", "arrest_record"],
    "edges": [
        ("race", "ses"),           # Race influences SES
        ("ses", "prior_arrests"),  # SES influences arrest patterns
        ("prior_arrests", "recidivism"),  # Prior arrests predict recidivism
        ("arrest_record", "recidivism"),  # Arrest record predicts outcome
    ],
    "non_causal_edges": [
        ("race", "recidivism"),  # Should NOT have direct causal link
    ]
}

# Analyze
analyzer = CausalAnalyzer(graph=causal_graph)
causal_report = analyzer.analyze(df_anon)

# Check for suspicious correlations
print("Suspicious Correlations:")
for susp in causal_report.suspicious_correlations:
    print(f"  {susp['pair']}: empirical_corr={susp['corr']:.3f}, "
          f"expected='none', status='FLAG FOR REVIEW'")

# Example output:
# race <-> recidivism: empirical_corr=0.28, status='FLAG FOR REVIEW'
# (Suggests possible omitted confounder: SES not fully captured)
```

**HITL Review Interface:**

```python
from fair_care.silver.human_in_loop import HITLDashboard

# Create HITL interface
hitl = HITLDashboard(
    causal_graph=causal_graph,
    suspicious_correlations=causal_report.suspicious_correlations,
    anonymization_report=anon_report,
    utility_metrics=utility_assess.metrics
)

# Expert review (in Streamlit UI)
# - Expert validates causal edges: "Approve", "Remove", or "Add new variable"
# - Expert assesses anonymization trade-off: "Accept k=5, DP ε=0.1"
# - Expert signs off: "Approved for Gold layer"

# Retrieve decisions
expert_decisions = hitl.get_decisions()
# Returns: {approved: True, modifications: [...], timestamp: ..., expert_id: ...}
```

#### 2c. Utility Assessment

```python
from fair_care.silver.utility_assessment import UtilityAssessment

utility_assess = UtilityAssessment(
    original_data=df_bronze,
    anonymized_data=df_anon
)

# Compute utility metrics
utility_report = utility_assess.compute_metrics()

print("Utility Assessment Report:")
print(f"  Correlation preservation (Hellinger): {utility_report.hellinger:.4f}")
print(f"  Predictive utility (AUROC delta): {utility_report.auroc_delta:.3f}")
print(f"  Overall utility score: {utility_report.utility_score:.3f}")

# Expected output:
# Correlation preservation: 0.0385 (good, <0.05)
# Predictive utility: 0.043 (excellent, original AUROC=0.71, anonymized=0.68)
# Overall utility score: 0.94
```

**Silver Layer Metrics:**

```python
from fair_care.metrics import SilverMetrics

metrics_s = SilverMetrics(
    anonymization_report=anon_report,
    utility_report=utility_report,
    causal_report=causal_report,
    hitl_decisions=expert_decisions
)

s_s = metrics_s.compute_score()
print(f"Anonymization Strength: {metrics_s.anonymization_strength:.3f}")  # 0.90
print(f"Utility Preservation: {metrics_s.utility_preservation:.3f}")      # 0.94
print(f"Causal Completeness: {metrics_s.causal_completeness:.3f}")        # 0.87
print(f"HITL Approval Rate: {metrics_s.hitl_approval_rate:.3f}")          # 0.92
print(f"Silver Score (S_S): {s_s:.3f}")                                   # 0.91
```

---

### Layer 3: Gold – Fairness & Feature Engineering

**Responsibility:** Apply bias mitigation, compute fairness metrics, prepare for ML deployment.

#### 3a. Bias Mitigation

```python
from aif360.algorithms.preprocessing import Reweighing
from fair_care.gold.bias_mitigation import BiasMitigator

# Define fairness target
fairness_config = {
    "protected_attribute": "race",
    "privileged_groups": [{"race": 1}],  # 1 = Caucasian in COMPAS
    "unprivileged_groups": [{"race": 0}],  # 0 = African American
    "target": "recidivism",
    "mitigation_technique": "reweighing"  # or "threshold_optimization", "disparate_impact_remover"
}

# Apply bias mitigation
mitigator = BiasMitigator(config=fairness_config)
df_gold, mitigation_report = mitigator.mitigate(df_anon)

print(f"Mitigation Report:")
print(f"  Weights applied: {mitigation_report.weights_summary}")
print(f"  Records reweighted: {mitigation_report.reweighted_count}")
```

#### 3b. Fairness Metrics Computation

```python
from fair_care.gold.fairness_metrics import FairnessMetrics

fairness_metrics = FairnessMetrics(
    dataset=df_gold,
    protected_attribute="race",
    target="recidivism",
    privileged_groups=[{"race": 1}],
    unprivileged_groups=[{"race": 0}]
)

# Compute fairness metrics
metrics_report = fairness_metrics.compute_all()

print("Fairness Metrics:")
print(f"  Demographic Parity Difference: {metrics_report.dpd:.4f}  (target: <0.10)")
print(f"  Equalized Odds Difference: {metrics_report.eod:.4f}  (target: <0.10)")
print(f"  Disparate Impact Ratio: {metrics_report.dir:.4f}  (target: >0.80)")
print(f"  Counterfactual Fairness: {metrics_report.cf:.4f}  (target: >0.90)")

# Status
dpd_status = "PASS" if metrics_report.dpd < 0.10 else "FLAG"
eod_status = "PASS" if metrics_report.eod < 0.10 else "FLAG"
print(f"\nStatus: DPD={dpd_status}, EOD={eod_status}")
```

#### 3c. Feature Engineering

```python
from fair_care.gold.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(
    categorical_features=["race", "gender", "employment"],
    numeric_features=["age", "prior_count"],
    target="recidivism"
)

# Generate features
df_features = engineer.engineer(df_gold)

# Compute feature quality
feature_quality = engineer.compute_quality_metrics()
print(f"Feature Completeness: {feature_quality.completeness:.3f}")  # % non-null
print(f"Feature Interpretability: {feature_quality.interpretability:.3f}")
print(f"Overall Feature Quality: {feature_quality.overall_score:.3f}")
```

#### 3d. Vector Embedding for RAG

```python
from fair_care.gold.embeddings import EmbeddingGenerator

# For text fields (if applicable)
generator = EmbeddingGenerator(
    model="all-MiniLM-L6-v2",  # Sentence transformer
    text_fields=["incident_description", "officer_notes"]
)

# Generate embeddings
embeddings = generator.generate(df_gold)

# Store in vector database
milvus_client = MilvusConnection(uri="http://milvus:19530")
milvus_client.insert(
    collection_name="compas_embeddings",
    data=embeddings,
    metadata={"fairness_status": "mitigated", "anonymized": True}
)
```

**Gold Layer Metrics:**

```python
from fair_care.metrics import GoldMetrics

metrics_g = GoldMetrics(
    fairness_report=metrics_report,
    feature_quality=feature_quality,
    utility_retention=0.91  # vs. original
)

s_g = metrics_g.compute_score()
print(f"Fairness Metrics Score: {metrics_g.fairness_score:.3f}")     # 0.85
print(f"Feature Quality: {metrics_g.feature_quality_score:.3f}")     # 0.88
print(f"Utility Retention: {metrics_g.utility_retention_score:.3f}") # 0.91
print(f"Gold Score (S_G): {s_g:.3f}")                               # 0.88
```

---

### Composite: FAIR-CARE Score

```python
from fair_care.metrics import FAIRCAREScore

# Aggregate scores
fair_care = FAIRCAREScore(
    s_b=0.95,
    s_s=0.91,
    s_g=0.88,
    weights={"w_B": 0.25, "w_S": 0.40, "w_G": 0.35}
)

score = fair_care.compute()
print(f"FAIR-CARE Score: {score:.3f}")  # 0.90

# Interpretation
if score >= 0.85:
    print("Status: EXCELLENT – Robust ethical governance")
elif score >= 0.70:
    print("Status: ACCEPTABLE – Targeted improvements recommended")
else:
    print("Status: AT RISK – Significant governance gaps")
```

---

## Algorithm Implementation Details

### Algorithm 1: PII Detection

**File:** `src/fair_care/bronze/pii_detection.py`

```python
class PIIDetection:
    def __init__(self, confidence_threshold=0.95, techniques=["regex", "nlp"]):
        self.threshold = confidence_threshold
        self.techniques = techniques
        self.patterns = self._load_patterns()
    
    def _load_patterns(self):
        return {
            "ssn": r"^\d{3}-\d{2}-\d{4}$",
            "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
            "phone": r"^\+?[\d\s\-\(\)]{10,}$",
            # ... 15 more HIPAA patterns
        }
    
    def detect(self, df):
        """
        Input: Spark DataFrame
        Output: {column: [pii_types], confidence: [scores]}
        """
        pii_report = {}
        
        for col in df.columns:
            sample = df.select(col).limit(1000).collect()
            matches = {}
            
            # Pattern matching
            for pattern_type, pattern in self.patterns.items():
                match_count = sum(1 for row in sample if regex.match(pattern, str(row[col])))
                match_rate = match_count / len(sample)
                if match_rate > 0.8:
                    matches[pattern_type] = match_rate
            
            # NLP-based detection (spaCy NER)
            if "nlp" in self.techniques:
                nlp = spacy.load("en_core_web_sm")
                doc_texts = [str(row[col]) for row in sample]
                for doc in nlp.pipe(doc_texts):
                    for ent in doc.ents:
                        if ent.label_ in ["PERSON", "GPE", "ORG"]:
                            matches["nlp_entity"] = matches.get("nlp_entity", 0) + 1
            
            if matches:
                pii_report[col] = {
                    "pii_types": list(matches.keys()),
                    "confidence": list(matches.values()),
                    "recommendation": "REMOVE"  # or "QUASI-IDENTIFIER"
                }
        
        return pii_report
```

**Testing:**

```python
# tests/test_pii_detection.py
def test_pii_detection():
    detector = PIIDetection()
    
    # Create test data
    test_df = spark.createDataFrame([
        ("John Doe", "123-45-6789", "john@example.com"),
        ("Jane Smith", "987-65-4321", "jane@example.com"),
    ], ["name", "ssn", "email"])
    
    report = detector.detect(test_df)
    
    assert "name" in report
    assert "ssn" in report["ssn"]["pii_types"]
    assert "email" in report["email"]["pii_types"]
    assert report["ssn"]["confidence"][0] > 0.90  # High confidence
    
    print("✓ PII detection test passed")
```

---

### Algorithm 2: Anonymization with Utility Assessment

**File:** `src/fair_care/silver/anonymization.py`

```python
class AnonymizationEngine:
    def anonymize(self, df_bronze, config):
        """
        Apply anonymization technique per config
        
        Args:
            df_bronze: Spark DataFrame (raw data)
            config: {technique, k_value, epsilon, quasi_identifiers, ...}
        
        Returns:
            (df_anon, report)
        """
        
        # Step 1: Convert to ARX format
        data_pandas = self._spark_to_pandas(df_bronze)
        
        # Step 2: Apply anonymization
        if config["technique"] == "k_anonymity":
            result = self._k_anonymity(data_pandas, config)
        elif config["technique"] == "differential_privacy":
            result = self._differential_privacy(data_pandas, config)
        elif config["technique"] == "l_diversity":
            result = self._l_diversity(data_pandas, config)
        else:
            raise ValueError(f"Unknown technique: {config['technique']}")
        
        df_anon = result.anonymized_data
        
        # Step 3: Assess utility
        utility_metrics = self._assess_utility(df_bronze, df_anon)
        
        # Step 4: Compile report
        report = AnonymizationReport(
            technique=config["technique"],
            k_achieved=result.k_value,
            suppressed_count=result.suppressed_count,
            info_loss=result.information_loss,
            utility_score=utility_metrics["overall"],
            privacy_risk=result.privacy_risk
        )
        
        return df_anon, report
    
    def _k_anonymity(self, data, config):
        """Apply k-anonymity via ARX"""
        import arx4py
        
        arx_config = arx4py.ARXConfiguration()
        arx_config.addPrivacyModel(arx4py.KAnonymity(config["k_value"]))
        
        # Mark quasi-identifiers
        for qi in config["quasi_identifiers"]:
            arx_config.addAttribute(qi, attribute_type="quasi-identifying")
        
        # Mark sensitive attributes
        for sa in config["sensitive_attributes"]:
            arx_config.addAttribute(sa, attribute_type="sensitive")
        
        result = arx4py.anonymize(data, arx_config)
        return result
    
    def _differential_privacy(self, data, config):
        """Apply differential privacy"""
        import diffprivlib.models as dp_models
        
        # Mechanism: Laplace or Gaussian noise
        epsilon = config["epsilon"]
        delta = config.get("delta", 1e-6)
        
        data_dp = data.copy()
        for col in config["quasi_identifiers"]:
            if data[col].dtype in ['float64', 'int64']:
                # Add Laplace noise
                scale = (data[col].max() - data[col].min()) / epsilon
                noise = np.random.laplace(0, scale, len(data))
                data_dp[col] = data[col] + noise
        
        return PrivacyResult(
            anonymized_data=data_dp,
            privacy_risk=self._estimate_privacy_risk_dp(epsilon),
            technique="differential_privacy"
        )
    
    def _assess_utility(self, original, anonymized):
        """Measure utility preservation"""
        # Correlation distance
        corr_orig = original.corr()
        corr_anon = anonymized.corr()
        hellinger = self._hellinger_distance(corr_orig.values, corr_anon.values)
        
        # Predictive utility: train simple model on both
        from sklearn.linear_model import LogisticRegression
        
        features = [...available features...]
        target = "target_column"
        
        model_orig = LogisticRegression().fit(original[features], original[target])
        model_anon = LogisticRegression().fit(anonymized[features], anonymized[target])
        
        auroc_orig = roc_auc_score(original[target], model_orig.predict_proba(original[features])[:, 1])
        auroc_anon = roc_auc_score(anonymized[target], model_anon.predict_proba(anonymized[features])[:, 1])
        
        pred_utility = 1 - abs(auroc_orig - auroc_anon)
        
        return {
            "correlation_preservation": 1 - hellinger,
            "predictive_utility": pred_utility,
            "overall": (1 - hellinger) * 0.5 + pred_utility * 0.5
        }
```

---

## Data Pipeline Walkthrough

### End-to-End Example: COMPAS Dataset

```bash
# 1. Download COMPAS data
python scripts/download_datasets.py --datasets compas

# 2. Inspect raw data
jupyter notebook notebooks/01_data_preparation.ipynb
# Output: 6,214 records, 13 features, bias baseline computed

# 3. Run full pipeline
python -m fair_care.orchestration.pipeline \
  --dataset compas \
  --config configs/default.yaml \
  --output results/compas_run001 \
  --verbose

# 4. Examine results
cat results/compas_run001/metrics_summary.json
# {
#   "fair_care_score": 0.80,
#   "s_b": 0.95,
#   "s_s": 0.91,
#   "s_g": 0.88,
#   "dpd": 0.12,
#   "eod": 0.16,
#   "utility_retention": 0.91,
#   "privacy_risk": 0.03
# }

# 5. View outputs
ls results/compas_run001/
# bronze/              (raw ingested data)
# silver/              (anonymized data)
# gold/                (fair, feature-ready data)
# metrics/             (all computed metrics)
# lineage/             (data provenance)
# hitl_decisions.log   (expert review log)
```

### Manual Step-by-Step

```python
from fair_care.orchestration.pipeline import FAIRCAREPipeline

# Initialize pipeline
pipeline = FAIRCAREPipeline(
    dataset_name="compas",
    config_file="configs/default.yaml"
)

# Bronze layer
print("🔵 BRONZE LAYER: Ingesting raw data...")
df_bronze = pipeline.bronze.ingest()
print(f"  → Ingested {df_bronze.count()} records")
print(f"  → PII fields detected: {pipeline.bronze.pii_detected}")
print(f"  → Provenance metadata: {pipeline.bronze.metadata}")

# Silver layer
print("\n🟣 SILVER LAYER: Anonymizing and validating...")
df_silver, silver_metrics = pipeline.silver.process(df_bronze)
print(f"  → Anonymization technique: {silver_metrics.technique}")
print(f"  → K-value achieved: {silver_metrics.k_value}")
print(f"  → Utility score: {silver_metrics.utility:.3f}")
print(f"  → Causal validation: APPROVED")

# Gold layer
print("\n🟡 GOLD LAYER: Applying fairness and preparing features...")
df_gold, gold_metrics = pipeline.gold.process(df_silver)
print(f"  → Demographic Parity Diff: {gold_metrics.dpd:.4f}")
print(f"  → Equalized Odds Diff: {gold_metrics.eod:.4f}")
print(f"  → Feature quality score: {gold_metrics.feature_quality:.3f}")

# Composite scoring
print("\n📊 COMPOSITE SCORE...")
fair_care_score = pipeline.compute_fair_care_score()
print(f"  → FAIR-CARE Score: {fair_care_score.score:.3f}")
print(f"  → Status: {fair_care_score.status}")

# Save and audit
pipeline.save_all(output_dir="results/compas_run001")
print("\n✅ Pipeline complete. Results saved to results/compas_run001/")
```

---

## Experiment Reproduction

### Experiment 1: Ablation Study

```bash
# Run all configurations on COMPAS dataset
python experiments/scripts/run_experiment_1.py \
  --dataset compas \
  --configs baseline,config_a,config_b,config_c \
  --output results/exp1.csv \
  --num_runs 3

# Expected runtime: ~30 minutes
# Output: results/exp1.csv with columns:
# config, s_b, s_s, s_g, fair_care_score, dpd, eod, utility, privacy_risk
```

### Experiment 2: Multi-Dataset Benchmarking

```bash
# Run on all three datasets
python experiments/scripts/run_experiment_2.py \
  --datasets compas,adult,german_credit \
  --config configs/default.yaml \
  --output results/exp2.csv \
  --parallel

# Expected runtime: ~45 minutes (parallelized)
# Output: results/exp2.csv with results for all datasets
```

### Experiment 3: Regulatory Compliance

```bash
# Test GDPR, HIPAA, CCPA configurations
python experiments/scripts/run_experiment_3.py \
  --datasets compas,adult,german_credit \
  --regulations gdpr,hipaa,ccpa \
  --output results/exp3.csv

# Expected runtime: ~60 minutes
# Output: results/exp3.csv with compliance scores per regulation
```

### Aggregate and Visualize Results

```bash
# Generate plots and summary statistics
python experiments/scripts/aggregate_results.py \
  --exp1 results/exp1.csv \
  --exp2 results/exp2.csv \
  --exp3 results/exp3.csv \
  --output_dir results/figures/

# Output:
# results/figures/ablation_study.png
# results/figures/benchmarking.png
# results/figures/compliance_comparison.png
```

---

## Evaluation Metrics Explained

### Bronze Layer (S_B)

**Provenance Score:**
- Measures completeness of metadata (ingestion timestamp, source system, steward, checksums)
- Target: 100% of records with complete provenance
- Formula: `(records_with_provenance / total_records) * 100`

**PII Detection Rate:**
- Precision & recall of PII detection against ground truth
- Target: 95%+ accuracy
- Formula: `(true_positives / (true_positives + false_positives)) * 0.5 + (true_positives / (true_positives + false_negatives)) * 0.5`

**Quality Baseline Score:**
- Data quality checks: null %, duplicates %, schema validity
- Target: >90%
- Formula: `(1 - null_pct) * 0.4 + (1 - duplicate_pct) * 0.3 + schema_validity * 0.3`

### Silver Layer (S_S)

**Anonymization Strength:**
- Normalized privacy guarantee (k-value, ε, equivalence class size)
- Target: 0.85+ (strong privacy)
- Formula (k-anonymity): `min(k, threshold_k) / threshold_k` (e.g., if k=5 and threshold=10, score=0.5)
- Formula (DP): `1 - (epsilon / epsilon_max)` (e.g., if ε=0.1 and max=1.0, score=0.9)

**Utility Preservation:**
- Correlation preservation + predictive utility balance
- Target: >85%
- Formula: `correlation_preservation * 0.5 + predictive_utility * 0.5`

**Causal Completeness:**
- % of critical causal pathways validated by domain experts
- Target: 100%
- Formula: `validated_pathways / total_pathways`

**HITL Approval Rate:**
- % of Silver layer outputs approved without requesting iteration
- Target: 80%+
- Formula: `approved_decisions / total_decisions`

### Gold Layer (S_G)

**Fairness Metrics Score:**
- Aggregated fairness metric performance (DPD, EOD, DI, CF)
- Target: 100% metrics passing targets
- Formula: `metrics_passing_target / total_metrics`

**Feature Quality:**
- Feature completeness, cardinality, interpretability
- Target: >90%
- Formula: `(1 - null_pct) * 0.4 + cardinality_score * 0.3 + interpretability_score * 0.3`

**Utility Retention:**
- Predictive performance compared to original data
- Target: >95% of original performance
- Formula: `(gold_auroc / original_auroc) * 100`

### Composite: FAIR-CARE Score

**Definition:**
$$FAIR\text{-}CARE\_Score = w_B \cdot S_B + w_S \cdot S_S + w_G \cdot S_G$$

**Default weights (balanced):**
- $w_B = 0.25$ (ingestion quality)
- $w_S = 0.40$ (anonymization & causality – most critical)
- $w_G = 0.35$ (fairness & utility)

**Interpretation:**
- **≥0.85:** EXCELLENT – Robust ethical governance, ready for deployment
- **0.70–0.85:** ACCEPTABLE – Governance in place, targeted improvements recommended
- **<0.70:** AT RISK – Significant governance gaps, architectural review needed

---

## Troubleshooting and FAQs

### Q1: Pipeline fails at Bronze layer with "PII detection timeout"

**Solution:**
```python
# Reduce sample size for PII detection
config = {
    "pii_detection": {
        "sample_size": 500,  # Default: 1000
        "confidence_threshold": 0.90  # Slightly lower threshold
    }
}
```

### Q2: Silver layer anonymization produces very high information loss (>0.20)

**Diagnosis & Solution:**
```python
# Try different anonymization technique
# If k-anonymity causes high loss, switch to DP with higher epsilon

config_original = {"technique": "k_anonymity", "k_value": 5}
config_improved = {"technique": "differential_privacy", "epsilon": 0.5}

# Test both and compare utility scores
# DP typically preserves utility better for moderate privacy levels
```

### Q3: Causal graph validation returns many "suspicious correlations"

**Interpretation:**

High correlation between variables marked as non-causal can indicate:
1. **Omitted confounder:** Missing variable that explains the correlation
2. **Measurement artifact:** Data quality issue, selection bias
3. **True causal link:** Domain assumption incorrect; requires expert revision

**Action:**
```python
# In HITL interface, for each suspicious correlation:
# - Review correlation strength and significance
# - Consult domain literature or expert
# - Update causal graph accordingly
# - Document reasoning for audit trail

# Example:
# Correlation: race <-> arrest_record (r=0.32, p<0.001)
# Possible confounder: neighborhood_crime_rate
# Action: Add node, update edges
```

### Q4: FAIR-CARE Score plateaus at 0.75; how to improve?

**Diagnostic:**
```python
# Check which component is bottleneck
# If S_S (Silver) is low:
#   → Improve causal validation (add missing variables)
#   → Try different anonymization technique
# If S_G (Gold) is low:
#   → Strengthen fairness metric targets
#   → Examine data for residual bias

# Use weighted adjustment based on priorities
config = {
    "weights": {
        "w_B": 0.20,
        "w_S": 0.20,
        "w_G": 0.60  # Emphasize fairness if that's bottleneck
    }
}
```

### Q5: Regulatory compliance fails for GDPR; privacy risk >2%

**Solution:**
```python
# Tighten anonymization for GDPR
config_gdpr = {
    "anonymization": {
        "technique": "differential_privacy",
        "epsilon": 0.05,  # Stricter: was 0.10
        "delta": 1e-7     # Tighter: was 1e-6
    }
}

# Expected result:
# Privacy risk drops to <0.02
# Utility retention may drop 5-10% (acceptable trade-off for GDPR compliance)
```

### Q6: How to adapt pipeline for my custom dataset?

**Steps:**
1. **Ingest data:**
   ```python
   from fair_care.bronze import DataIngestion
   
   ingestion = DataIngestion(
       source="my_dataset.csv",
       source_system="My Organization",
       steward_contact="data@myorg.org"
   )
   df = ingestion.ingest()
   ```

2. **Identify quasi-identifiers and sensitive attributes:**
   ```python
   config = {
       "quasi_identifiers": ["age", "postcode", "occupation"],
       "sensitive_attributes": ["health_status", "financial_score"]
   }
   ```

3. **Elicit causal assumptions from domain experts:**
   ```python
   causal_graph = {
       "nodes": [...],
       "edges": [("variable_A", "variable_B"), ...]
   }
   ```

4. **Run pipeline:**
   ```python
   pipeline = FAIRCAREPipeline(dataset_name="my_dataset", config=config)
   result = pipeline.run_full()
   ```

### Q7: Can I use FAIR-CARE for real-time data streams?

**Current:** Designed for batch processing (full dataset anonymization + fairness evaluation).

**Future roadmap:** Federated FAIR-CARE for streaming data (Part of future work in paper Section VIII-B).

For now, apply FAIR-CARE in periodic batch cycles (e.g., nightly) and monitor fairness metrics on streamed predictions.

---

## Additional Resources

- **Paper:** See FAIR-CARE-Lakehouse-Paper.md for full methodology and experiments
- **API Documentation:** docs/api.md
- **Configuration Guide:** docs/configuration.md
- **Related Papers:**
  - Dressel & Farid (2018): "The accuracy, fairness, and limits of predicting recidivism"
  - Prasser & Kohlmayer (2015): "ARX: An extensible toolkit for anonymization" 
  - Bellamy et al. (2019): "AI Fairness 360"
- **Public Datasets:**
  - COMPAS: https://www.kaggle.com/datasets/danofer/compass
  - Adult Census: https://archive.ics.uci.edu/ml/datasets/adult
  - German Credit: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

---

**Last Updated:** 2025-01-15  
**Maintainers:** [Research Team]  
**License:** Apache 2.0

