class BronzeMetrics:
    def calculate(self, metadata: dict) -> float:
        # SB = w1*Provenance + w2*PII + w3*Quality
        # Simplified
        provenance_score = 1.0 if metadata.get("provenance_complete") else 0.5
        pii_score = 1.0 if not metadata.get("pii_found") else 0.0
        quality_score = metadata.get("quality_score", 0.8)
        
        return (provenance_score + pii_score + quality_score) / 3

class SilverMetrics:
    def calculate(self, metadata: dict) -> float:
        # SS = w1*Anonymization + w2*Utility + w3*Causal
        anon_score = 1.0 # Assumed if process ran
        utility_score = metadata.get("utility_retention", 0.0)
        causal_score = 1.0 if metadata.get("causal_validity") == "PASS" else 0.5
        
        return (anon_score + utility_score + causal_score) / 3

class GoldMetrics:
    def calculate(self, metadata: dict) -> float:
        # SG = w1*Fairness + w2*FeatureQuality + w3*UtilityRetention
        spd = metadata.get("statistical_parity_difference")
        if spd is None: spd = 1.0
        fairness_score = 1.0 if abs(spd) < 0.1 else 0.0
        feature_score = 0.9 # Placeholder
        utility_score = metadata.get("utility_retention", 0.0) # From Silver/Gold comparison
        
        return (fairness_score + feature_score + utility_score) / 3
