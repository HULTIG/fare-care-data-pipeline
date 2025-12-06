import pandas as pd
from pyspark.sql import DataFrame
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

class FairnessMetrics:
    def __init__(self, config: dict):
        self.config = config

    def calculate(self, df: DataFrame) -> dict:
        """
        Calculates fairness metrics.
        """
        print("Calculating Fairness Metrics...")
        pdf = df.toPandas()
        
        prot_attr = self.config.get("protected_attribute")
        label_col = self.config.get("label_column")
        priv_group = self.config.get("privileged_groups", [{}])[0]
        unpriv_group = self.config.get("unprivileged_groups", [{}])[0]
        
        report = {}
        
        if prot_attr and label_col:
            try:
                dataset = BinaryLabelDataset(
                    favorable_label=self.config.get("favorable_label", 1),
                    unfavorable_label=0 if self.config.get("favorable_label") == 1 else 1,
                    df=pdf,
                    label_names=[label_col],
                    protected_attribute_names=[prot_attr]
                )
                
                metric = BinaryLabelDatasetMetric(
                    dataset,
                    unprivileged_groups=[unpriv_group],
                    privileged_groups=[priv_group]
                )
                
                report["statistical_parity_difference"] = metric.statistical_parity_difference()
                report["disparate_impact"] = metric.disparate_impact()
                # EOD requires predictions, skipping for dataset-only metrics or implementing if needed
                
            except Exception as e:
                print(f"Fairness metrics calculation failed: {e}")
                report["error"] = str(e)
                
        print(f"Fairness Metrics: {report}")
        return report
