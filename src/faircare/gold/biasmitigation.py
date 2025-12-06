import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

class BiasMitigator:
    def __init__(self, config: dict):
        self.config = config

    def mitigate(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Mitigates bias using Reweighing.
        """
        print("Running Bias Mitigation...")
        pdf = df.toPandas()
        
        prot_attr = self.config.get("protected_attribute")
        label_col = self.config.get("label_column")
        priv_group = self.config.get("privileged_groups", [{}])[0]
        unpriv_group = self.config.get("unprivileged_groups", [{}])[0]
        
        if not (prot_attr and label_col):
            print("Missing config for bias mitigation. Skipping.")
            return df

        # Prepare data for AIF360
        # Need to ensure numeric encoding for AIF360
        # Simplified handling:
        try:
            dataset = BinaryLabelDataset(
                favorable_label=self.config.get("favorable_label", 1),
                unfavorable_label=0 if self.config.get("favorable_label") == 1 else 1, # Assumption
                df=pdf,
                label_names=[label_col],
                protected_attribute_names=[prot_attr]
            )
            
            RW = Reweighing(
                unprivileged_groups=[unpriv_group],
                privileged_groups=[priv_group]
            )
            
            dataset_transf = RW.fit_transform(dataset)
            
            # Add weights back to dataframe
            pdf['instance_weights'] = dataset_transf.instance_weights
            
            print("Bias mitigation complete. Weights added.")
            return spark.createDataFrame(pdf)
            
        except Exception as e:
            print(f"Bias mitigation failed: {e}")
            return df
