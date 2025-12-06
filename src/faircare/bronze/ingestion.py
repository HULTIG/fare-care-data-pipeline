from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, input_file_name
import hashlib

class DataIngestion:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def ingest(self, source_path: str, output_path: str, dataset_name: str, source_system: str = "manual_upload"):
        """
        Ingests a CSV file into a Bronze Delta table.
        """
        print(f"Ingesting {dataset_name} from {source_path} to {output_path}...")
        
        # Read CSV
        # Using inferSchema for now, but in production we should enforce schema
        df = self.spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(source_path)

        # Add metadata columns
        df_with_meta = df \
            .withColumn("_ingestion_timestamp", current_timestamp()) \
            .withColumn("_source_system", lit(source_system)) \
            .withColumn("_source_file", input_file_name()) \
            .withColumn("_dataset_name", lit(dataset_name))

        # Calculate schema hash
        schema_str = str(df.schema)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
        df_with_meta = df_with_meta.withColumn("_schema_hash", lit(schema_hash))

        # Write to Delta
        df_with_meta.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(output_path)
            
        print(f"Ingestion complete. Count: {df_with_meta.count()}")
        return df_with_meta
