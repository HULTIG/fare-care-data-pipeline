# FAIR-CARE Lakehouse: Project Timeline & ICSA 2026 Submission Checklist

---

## Project Development Timeline (24 weeks)

### Phase 1: Foundation & Infrastructure (Weeks 1–4)

**Objective:** Establish development environment and implement Bronze layer

**Week 1: Setup & Data Preparation**
- [ ] Clone repositories (Spark, Delta, ARX)
- [ ] Set up GitHub repo with CI/CD pipeline
- [ ] Download and validate benchmark datasets (COMPAS, Adult Census, German Credit)
- [ ] Create development documentation (Architecture overview, file structure)
- **Deliverables:** GitHub repo, Docker setup, datasets validated

**Week 2: Bronze Layer – Data Ingestion**
- [ ] Implement DataIngestion class (source connectors, schema inference)
- [ ] Build provenance tracking module (checksums, metadata storage)
- [ ] Create unit tests for ingestion (5+ test cases)
- [ ] Document API: `df = DataIngestion(source).ingest()`
- **Deliverables:** Ingestion module working on all 3 datasets

**Week 3: PII Detection (Algorithm 1)**
- [ ] Implement regex-based patterns (18 HIPAA identifiers)
- [ ] Integrate spaCy NER for NLP-based detection
- [ ] Validate against known PII (95%+ accuracy target)
- [ ] Create test suite with synthetic PII data
- **Deliverables:** PII detector with 95%+ F1 score

**Week 4: Bias Audit & Bronze Metrics**
- [ ] Implement BiasAudit (demographic distribution computation)
- [ ] Calculate S_B (Bronze Score) metric
- [ ] Write unit tests (10+ test cases)
- [ ] Document bias audit output format
- **Deliverables:** Bronze layer complete, ready for Silver integration

**Milestone:** Bronze layer tested end-to-end on all 3 datasets ✓

---

### Phase 2: Silver Layer – Anonymization & Causality (Weeks 5–9)

**Objective:** Implement anonymization, causal analysis, and human-in-the-loop review

**Week 5: ARX Integration & Anonymization**
- [ ] Wrap ARX Python API (k-anonymity, l-diversity, t-closeness)
- [ ] Implement differential privacy integration (using diffprivlib)
- [ ] Create anonymization configuration builder
- [ ] Test all techniques on COMPAS dataset
- **Deliverables:** AnonymizationEngine class operational for all 4 techniques

**Week 6: Utility Assessment Module**
- [ ] Implement correlation preservation metrics (Hellinger distance)
- [ ] Build predictive utility measurement (train/test AUROC comparison)
- [ ] Integrate information loss metrics (NCP from ARX)
- [ ] Create UtilityAssessment class with comprehensive reporting
- **Deliverables:** Utility module with 3+ metrics, validated against baselines

**Week 7: Causal Graph Analysis (Algorithm 3)**
- [ ] Build CausalAnalyzer class (graph specification, correlation detection)
- [ ] Implement backdoor criterion checking (DoWhy integration)
- [ ] Create suspicious correlation flagging logic
- [ ] Design causal graph visualization (networkx → matplotlib/plotly)
- **Deliverables:** Causal analyzer working, visualization dashboard prototype

**Week 8: Human-in-the-Loop Interface**
- [ ] Design HITL UI (Streamlit mockups)
- [ ] Implement backend (decision storage, audit trail)
- [ ] Build causal graph validator (approve/reject edges, add notes)
- [ ] Build anonymization trade-off reviewer
- **Deliverables:** Functional HITL dashboard (Streamlit), expert review workflow

**Week 9: Silver Metrics & Integration**
- [ ] Calculate S_S (Silver Score) from all components
- [ ] Integrate HITL decisions into pipeline
- [ ] Write integration tests (5+ scenarios)
- [ ] Document Silver layer API and configuration
- **Deliverables:** Silver layer complete and tested; ready for Gold integration

**Milestone:** Silver layer end-to-end on all 3 datasets, HITL working ✓

---

### Phase 3: Gold Layer – Fairness & Features (Weeks 10–13)

**Objective:** Implement bias mitigation, fairness metrics, and feature engineering

**Week 10: Fairness Metrics Implementation**
- [ ] Integrate AIF360 fairness metrics (DPD, EOD, DI, counterfactual fairness)
- [ ] Implement Fairness Metrics class with multi-group support
- [ ] Create fairness report generation (JSON, CSV, HTML)
- [ ] Validate metrics against AIF360 benchmarks
- **Deliverables:** FairnessMetrics class with 4+ metrics, validated accuracy

**Week 11: Bias Mitigation Algorithms**
- [ ] Integrate AIF360 bias mitigation (reweighing, threshold optimization)
- [ ] Implement BiasMitigator class
- [ ] Apply mitigation on all 3 datasets
- [ ] Measure fairness improvement (target: >30% reduction in DPD)
- **Deliverables:** BiasMitigator operational, baseline fairness improvement validated

**Week 12: Feature Engineering & Quality**
- [ ] Implement feature importance computation (SHAP, tree-based)
- [ ] Build feature quality assessment (completeness, cardinality, interpretability)
- [ ] Create FeatureEngineer class for categorical encoding and scaling
- [ ] Design feature quality report
- **Deliverables:** FeatureEngineer class with quality scoring

**Week 13: Vector Embeddings & Gold Metrics**
- [ ] Integrate sentence transformers for text embedding
- [ ] Set up Milvus or Pinecone connection
- [ ] Create embedding generator and storage module
- [ ] Calculate S_G (Gold Score) from fairness, quality, and utility metrics
- [ ] Document Gold layer API
- **Deliverables:** Gold layer complete; vector embeddings working

**Milestone:** Gold layer end-to-end on all 3 datasets ✓

---

### Phase 4: Composite Metrics & Configuration (Weeks 14–16)

**Objective:** Implement FAIR-CARE Score and regulatory configurations

**Week 14: FAIR-CARE Score Implementation**
- [ ] Implement FAIRCAREScore class with weighting system
- [ ] Create interpretive guidance (≥0.85 = EXCELLENT, etc.)
- [ ] Build score dashboard visualization
- [ ] Write unit tests (10+ scenarios with different weights)
- **Deliverables:** FAIR-CARE Score fully functional, dashboard updated

**Week 15: Regulatory Compliance Mapping**
- [ ] Create compliance checker for GDPR, HIPAA, CCPA/CPRA
- [ ] Develop configuration templates for each regulation
- [ ] Implement compliance scoring logic
- [ ] Document mapping of techniques to regulatory standards
- **Deliverables:** 3 regulatory templates (gdpr.yaml, hipaa.yaml, ccpa.yaml)

**Week 16: Pipeline Orchestration & Logging**
- [ ] Build FAIRCAREPipeline orchestration class
- [ ] Implement end-to-end pipeline runner
- [ ] Set up comprehensive audit logging (all decisions, metrics, lineage)
- [ ] Create pipeline configuration system
- **Deliverables:** Full pipeline orchestration working; audit trails complete

**Milestone:** Composite metrics and regulatory compliance fully implemented ✓

---

### Phase 5: Experimentation & Evaluation (Weeks 17–20)

**Objective:** Execute experiments and validate research contributions

**Week 17: Experiment 1 – Ablation Study**
- [ ] Run baseline vs. Configs A, B, C on COMPAS
- [ ] Collect metrics: S_B, S_S, S_G, FAIR-CARE Score, DPD, EOD, Utility
- [ ] Analyze and document results
- [ ] Generate visualization: Ablation study plots
- **Deliverables:** Experiment 1 complete; results in results/exp1.csv

**Week 18: Experiment 2 – Multi-Dataset Benchmarking**
- [ ] Run full pipeline on COMPAS, Adult Census, German Credit
- [ ] Collect all metrics for all 3 datasets
- [ ] Compute statistics (mean, std, min, max across datasets)
- [ ] Identify dataset-specific challenges and successes
- **Deliverables:** Experiment 2 complete; results in results/exp2.csv

**Week 19: Experiment 3 – Regulatory Compliance**
- [ ] Run GDPR configuration on all datasets
- [ ] Run HIPAA configuration on all datasets
- [ ] Run CCPA/CPRA configuration on all datasets
- [ ] Verify compliance scores and achievable FAIR-CARE Scores
- [ ] Document regulatory mapping details
- **Deliverables:** Experiment 3 complete; results in results/exp3.csv

**Week 20: Results Aggregation & Visualization**
- [ ] Aggregate results from all experiments
- [ ] Generate publication-quality plots and tables
- [ ] Compute statistical significance (if applicable)
- [ ] Create summary tables for paper
- **Deliverables:** All figures and tables for paper; results/ directory finalized

**Milestone:** All experiments complete and validated ✓

---

### Phase 6: Paper Writing & Artifact Finalization (Weeks 21–24)

**Objective:** Write paper and prepare artifact for submission

**Week 21: Paper Drafting – Core Sections**
- [ ] Write Introduction & Motivation (2 pages)
- [ ] Write Background & Related Work (2 pages)
- [ ] Write Architecture & Methodology (3 pages)
- [ ] Compile experimental results into sections
- **Deliverables:** First draft of core content

**Week 22: Paper Finalization**
- [ ] Write Evaluation section with full results
- [ ] Write Discussion & Limitations
- [ ] Write Conclusion & Future Work
- [ ] Compile References (30–40 citations)
- [ ] Format according to IEEE template (10 pages + 2 ref pages)
- **Deliverables:** Complete paper draft

**Week 23: Artifact Preparation**
- [ ] Review artifact structure against submission guidelines
- [ ] Ensure code is well-commented and documented
- [ ] Create detailed README with setup, usage, and reproduction instructions
- [ ] Write Jupyter notebooks demonstrating each phase (01–07)
- [ ] Verify reproducibility: run artifact from scratch, confirm results match paper
- [ ] Create demo video (5–10 minutes)
- **Deliverables:** Anonymized artifact ready for submission

**Week 24: Submission & Final Checks**
- [ ] Anonymize author information in paper and artifact
- [ ] Double-check double-blind requirements
- [ ] Verify all figures and tables are clear and publication-ready
- [ ] Proofread paper (grammar, formatting, citations)
- [ ] Submit to EasyChair with all required documents
- [ ] Prepare supplementary materials (if allowed)
- **Deliverables:** Paper submitted to ICSA 2026 ✓

**Milestone:** Paper submitted and artifact ready for review ✓

---

## ICSA 2026 Submission Checklist

### Paper Requirements

**Content & Scope**
- [ ] Paper addresses ICSA 2026 main theme: "Architecting in Continuous Software Engineering: Evolving Roles, Enduring Principles"
- [ ] Paper is within scope: Software Architecture, Quality Attributes, AI/ML Systems, or Ethics
- [ ] Main text: 10 pages maximum (inclusive of figures and tables)
- [ ] References: 2 additional pages maximum
- [ ] Follows IEEE paper formatting guidelines

**Technical Contribution**
- [ ] Clear research questions (RQ1, RQ2, RQ3) identified and addressed
- [ ] Novel algorithmic contribution (FAIR-CARE pipeline)
- [ ] Quantifiable evaluation method (FAIR-CARE Score)
- [ ] Empirical validation on multiple datasets
- [ ] Regulatory compliance demonstrated (GDPR, HIPAA, CCPA/CPRA)

**Writing & Presentation**
- [ ] Abstract: Clear, compelling, <150 words
- [ ] Introduction: Problem well-motivated, contributions stated upfront
- [ ] Related work: Comprehensive coverage of fairness, privacy, architecture literature
- [ ] Methodology: Clear explanation of algorithms and architecture
- [ ] Evaluation: Rigorous experiments with appropriate metrics
- [ ] Conclusion: Summarizes contributions and future work
- [ ] Citations: Consistent formatting, 30–40 high-quality references
- [ ] Figures & Tables: Clear labels, captions, and legends

**Anonymity (Double-Blind Review)**
- [ ] Author names, affiliations, and identifying information removed
- [ ] Acknowledgments section removed or anonymized
- [ ] Self-citations modified or removed (use "Author et al., XXXX" format)
- [ ] No identifying URLs or repository links (use "will be available" or anonymous repos)
- [ ] Paper header: "Anonymous submission to ICSA 2026"

---

### Artifact Requirements

**Code & Documentation**
- [ ] Well-documented source code (comments, docstrings for all classes/functions)
- [ ] README.md with:
  - [ ] High-level description of artifact
  - [ ] Requirements (Python version, dependencies, hardware)
  - [ ] Installation instructions (step-by-step, 10 minutes max)
  - [ ] Quick-start guide with example command
  - [ ] Detailed usage instructions
  - [ ] Troubleshooting section
- [ ] LICENSE file (Apache 2.0 or similar permissive license)
- [ ] CITATION.cff file for citation format

**Reproducibility**
- [ ] requirements.txt with pinned versions of all dependencies
- [ ] setup.py for installation as Python package
- [ ] Docker/docker-compose.yml for isolated environment
- [ ] Configuration files for all experiments (YAML or JSON)
- [ ] Scripts to reproduce Experiments 1, 2, 3 with single command
- [ ] Expected outputs documented (run time, output file sizes, result summaries)

**Data & Experiments**
- [ ] Instructions for downloading public benchmark datasets (with URLs)
- [ ] Sample dataset included OR clear instructions for generation
- [ ] Jupyter notebooks (07 total) demonstrating:
  - [ ] Data preparation
  - [ ] Bronze layer walkthrough
  - [ ] Silver layer anonymization
  - [ ] Causal validation
  - [ ] Gold layer fairness
  - [ ] FAIR-CARE score computation
  - [ ] Full experiment reproduction
- [ ] Results CSV files from Experiments 1–3
- [ ] Visualization scripts generating paper figures

**Testing**
- [ ] Unit tests for all major modules (50+ test cases)
- [ ] Integration tests for full pipeline
- [ ] All tests pass (no failures or warnings)
- [ ] Test execution command: `pytest tests/`

**Dashboard/UI**
- [ ] Streamlit app (or equivalent) for interactive exploration
- [ ] Dashboard displays metrics, causal graphs, fairness visualizations
- [ ] Instructions for launching dashboard
- [ ] Screenshot of dashboard in README

**Evaluation Claims**
- [ ] Artifact demonstrates all claims in paper
- [ ] Experiments can be reproduced, results match paper within ±5%
- [ ] Pipeline completes in documented time (≤60 minutes for full pipeline)
- [ ] Metrics computed correctly (validated against known benchmarks)

---

### File Structure

```
fair-care-lakehouse/
├── README.md                          # ✓ Main entry point
├── LICENSE                            # ✓ Apache 2.0
├── CITATION.cff                       # ✓ Citation metadata
├── .gitignore
├── requirements.txt                   # ✓ Dependencies
├── setup.py                           # ✓ Installation
│
├── docs/
│   ├── architecture.md               # ✓ Overview
│   ├── installation.md               # ✓ Setup details
│   ├── experiments.md                # ✓ How to reproduce
│   ├── configuration.md              # ✓ Configuration options
│   └── api.md                        # ✓ API documentation
│
├── src/
│   ├── fair_care/                   # ✓ Main package
│   │   ├── __init__.py
│   │   ├── bronze/                  # ✓ Ingestion + PII
│   │   ├── silver/                  # ✓ Anonymization + Causal
│   │   ├── gold/                    # ✓ Fairness + Features
│   │   ├── metrics/                 # ✓ Scoring
│   │   └── orchestration/           # ✓ Pipeline runner
│   │
│   └── dashboard/                   # ✓ Streamlit UI
│       ├── app.py
│       └── components/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb     # ✓ Data loading
│   ├── 02_bronze_ingestion.ipynb     # ✓ Bronze layer
│   ├── 03_silver_anonymization.ipynb # ✓ Anonymization
│   ├── 04_causal_validation.ipynb    # ✓ Causal analysis
│   ├── 05_gold_fairness.ipynb        # ✓ Fairness metrics
│   ├── 06_fair_care_score.ipynb      # ✓ Score computation
│   └── 07_experiments.ipynb          # ✓ Run all experiments
│
├── tests/
│   ├── test_pii_detection.py        # ✓ Unit tests
│   ├── test_anonymization.py        # ✓ Unit tests
│   ├── test_causal_analysis.py      # ✓ Unit tests
│   ├── test_fairness_metrics.py     # ✓ Unit tests
│   └── test_fair_care_score.py      # ✓ Unit tests
│
├── experiments/
│   ├── configs/
│   │   ├── default.yaml             # ✓ Default config
│   │   ├── baseline.yaml            # ✓ Baseline
│   │   ├── config_a.yaml            # ✓ K-anonymity
│   │   ├── config_b.yaml            # ✓ Diff Privacy
│   │   ├── gdpr_strict.yaml         # ✓ GDPR
│   │   ├── hipaa.yaml               # ✓ HIPAA
│   │   └── ccpa.yaml                # ✓ CCPA
│   │
│   ├── scripts/
│   │   ├── run_experiment_1.py      # ✓ Ablation study
│   │   ├── run_experiment_2.py      # ✓ Benchmarking
│   │   ├── run_experiment_3.py      # ✓ Compliance
│   │   └── aggregate_results.py     # ✓ Visualization
│   │
│   └── results/
│       ├── exp1.csv                  # ✓ Ablation results
│       ├── exp2.csv                  # ✓ Benchmark results
│       └── exp3.csv                  # ✓ Compliance results
│
├── data/
│   ├── raw/README.md                # ✓ Dataset instructions
│   ├── processed/                   # ✓ Sample outputs
│   └── synthetic/                   # ✓ Demo data
│
├── docker-compose.yml               # ✓ Full stack
└── .github/
    └── workflows/
        └── ci.yml                    # ✓ CI/CD pipeline
```

---

### Artifact Submission Process

1. **Anonymize Artifact**
   ```bash
   # Remove all author information
   find . -name "*.py" -exec sed -i 's/Author Name/[Anonymous]/g' {} \;
   find . -name "*.md" -exec sed -i 's/Author Name/[Anonymous]/g' {} \;
   find . -name "*.ipynb" -exec sed -i 's/Author Name/[Anonymous]/g' {} \;
   
   # Verify anonymity
   grep -r "yourname\|@youremail\|yourdomain" src/ docs/ notebooks/
   # Should return nothing
   ```

2. **Create Anonymous Repository**
   ```bash
   # Create GitHub repo with no author affiliation
   # Use anonymous email (e.g., research@anonymous.org)
   # Copy anonymized code to new repo
   # Create temporary access link for review committee
   ```

3. **Package Artifact**
   ```bash
   # Create tarball
   tar -czf fair-care-lakehouse.tar.gz fair-care-lakehouse/
   
   # Verify contents
   tar -tzf fair-care-lakehouse.tar.gz | head -20
   ```

4. **Verify Reproducibility (Self-Check)**
   ```bash
   # On clean system:
   mkdir /tmp/review
   cd /tmp/review
   tar -xzf fair-care-lakehouse.tar.gz
   cd fair-care-lakehouse
   
   # Follow README instructions exactly
   pip install -r requirements.txt
   python -m fair_care.orchestration.pipeline \
     --dataset compas \
     --config configs/default.yaml \
     --output results/test_run
   
   # Verify results match paper ±5%
   cat results/test_run/metrics_summary.json
   ```

5. **Submit to Conference**
   - [ ] Upload paper PDF (anonymized) to EasyChair
   - [ ] Upload artifact tarball to conference systems
   - [ ] Include README.txt with setup instructions
   - [ ] Provide anonymous URL to artifact repository (GitHub with temporary access)
   - [ ] Include expected runtime and hardware requirements
   - [ ] Add any supplementary materials (demo video, detailed results tables)

---

### Review Readiness Checklist

**For Reviewers/Evaluation Committee**

- [ ] Artifact can be extracted and installed in <30 minutes
- [ ] All dependencies are correctly listed and installable
- [ ] Quick-start command runs successfully without errors
- [ ] Notebooks execute from start to finish without user intervention
- [ ] Experiments reproduce paper results within ±5% tolerance
- [ ] Code is readable, well-commented, and follows Python conventions
- [ ] Dashboard launches successfully and displays metrics
- [ ] Documentation is clear and sufficient to understand and modify code
- [ ] Tests pass with >95% test coverage for core modules
- [ ] Results are reproducible on different systems (Linux, macOS, Windows)

---

### Timeline to Conference

**ICSA 2026 Key Dates (Estimated)**

| Date | Event | Action |
|------|-------|--------|
| Jan 15, 2026 | Paper Submission Deadline | Submit paper + artifact |
| Feb 15, 2026 | Initial Review Complete | Respond to reviewer comments |
| Mar 15, 2026 | Artifact Evaluation | Artifact evaluated by committee |
| Apr 1, 2026 | Acceptance Notification | Learn of paper acceptance |
| Apr 15–May 15 | Camera-Ready Revision | Finalize paper and artifact |
| Jun 15–19, 2026 | Conference (assumed) | Present paper + demonstrate artifact |

**Working Backward from Jan 15 Submission:**
- Week 24 (Jan 8): Final submission preparation and checks
- Week 23 (Jan 1): Artifact finalization and testing
- Week 22 (Dec 24): Paper completion and proofreading
- Week 21 (Dec 17): Paper drafting begins
- Weeks 17–20 (Nov 19–Dec 14): Experiments and evaluation
- Weeks 14–16 (Oct 29–Nov 18): Composite metrics and configuration
- Weeks 10–13 (Oct 1–29): Gold layer implementation
- Weeks 5–9 (Aug 24–Sep 28): Silver layer implementation
- Weeks 1–4 (Jul 27–Aug 23): Foundation and Bronze layer

---

## Key Success Criteria

### Research Quality
- [ ] Novel algorithmic framework (FAIR-CARE pipeline)
- [ ] Addresses real problem (fairness + privacy in critical systems)
- [ ] Rigorous evaluation on multiple datasets
- [ ] Answers three clear research questions
- [ ] Findings are statistically significant

### Implementation Quality
- [ ] Code is production-ready and well-documented
- [ ] Reproducible experiments with clear instructions
- [ ] Comprehensive test coverage (>80%)
- [ ] Handles edge cases and error conditions
- [ ] Performance acceptable (pipeline completes in <1 hour)

### Presentation Quality
- [ ] Paper is well-written and clearly structured
- [ ] Figures and tables support key points
- [ ] References are comprehensive and accurate
- [ ] Methodology is clearly explained
- [ ] Results are interpreted correctly

### Artifact Quality
- [ ] Easy to install and use (README is clear)
- [ ] Reproducible experiments (±5% tolerance)
- [ ] Interactive visualizations (dashboard works)
- [ ] Comprehensive documentation
- [ ] Follows conference submission guidelines

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| ARX integration bugs | Medium | High | Early integration testing; backup with k-anonymity fallback |
| Causal graph expert availability | High | Medium | Pre-specify graphs; use synthetic scenarios if experts unavailable |
| Fairness metric implementation errors | Low | High | Validate against AIF360 benchmarks; use published test datasets |
| Dataset download failures | Low | Medium | Host backup copies; document URLs; include sample data |
| Compute resource constraints | Low | Medium | Optimize for smaller samples; use cloud resources (AWS, GCP) if needed |
| Paper acceptance uncertainty | High | High | Publish preprint on arXiv; position as contribution to ICSA community regardless |

---

## Post-Submission (If Accepted)

- [ ] Prepare conference presentation (slides, demo video)
- [ ] Practice talk (aim for 15–20 minute slot)
- [ ] Prepare live demo (backup with recorded video)
- [ ] Register for conference
- [ ] Arrange travel if in-person attendance required
- [ ] Prepare extended version for journal submission (future work)
- [ ] Release code and artifact to public GitHub (non-anonymous, after review)

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-15  
**Status:** Ready for execution

