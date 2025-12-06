import pandas as pd
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from diffprivlib.mechanisms import Laplace

class AnonymizationEngine:
    def __init__(self, config: dict):
        self.config = config

    def anonymize(self, df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Anonymizes the input DataFrame based on configuration.
        """
        print("Running Anonymization...")
        pdf = df.toPandas()
        
        technique = self.config.get("technique", "kanonymity")
        
        if technique == "kanonymity":
            pdf = self._apply_kanonymity(pdf)
        elif technique == "differentialprivacy":
            pdf = self._apply_differential_privacy(pdf)
            
        # Convert back to Spark DataFrame
        # Handle potential type issues by converting objects to string if needed or inferSchema
        return spark.createDataFrame(pdf)

    def _apply_kanonymity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies basic k-anonymity by generalizing quasi-identifiers.
        This is a simplified implementation.
        """
        k = self.config.get("k", 5)
        qis = self.config.get("quasi_identifiers", [])
        
        print(f"Applying k-anonymity (k={k}) on {qis}...")
        
        # Simple generalization: binning numeric values, masking strings
        for col in qis:
            if col not in df.columns: continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Binning
                try:
                    df[col] = pd.cut(df[col], bins=10).astype(str)
                except:
                    pass
            else:
                # Masking last characters
                df[col] = df[col].astype(str).apply(lambda x: x[:-3] + "***" if len(x) > 3 else "*")
        
        # Suppression: Remove groups with size < k
        if qis:
            groups = df.groupby(qis)
            df = groups.filter(lambda x: len(x) >= k)
            
        print(f"Rows remaining after k-anonymity suppression: {len(df)}")
        return df

    def _apply_differential_privacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Differential Privacy using Laplace mechanism on numeric columns.
        """
        epsilon = self.config.get("epsilon", 1.0)
        print(f"Applying Differential Privacy (epsilon={epsilon})...")
        
        mech = Laplace(epsilon=epsilon, sensitivity=1)
        
        for col in df.columns:
            if col.startswith("_"): continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(mech.randomise)
                
        return df
