# Databricks notebook source
# Export the gold table as Parquet so models can be trained locally (one row per user per cutoff, ~2M rows each).
from pyspark.sql import functions as F

OUT = "/Volumes/workspace/ecom/raw/export/gold_user_features"

g = spark.table("workspace.ecom.gold_user_features")
display(g.groupBy("cutoff_date").count().orderBy("cutoff_date"))

# One file per cutoff: easy to download, and the folder name carries the cutoff date
g.repartition("cutoff_date").write.mode("overwrite").partitionBy("cutoff_date").parquet(OUT)
