# Databricks notebook source
# Bronze and silver: load REES46 monthly files into Delta, then clean them.
# Bronze = raw events with an explicit schema. Silver = deduplicated, invalid rows handled, data-quality summary.
from pyspark.sql import functions as F

MONTHS = ["2019-Oct", "2019-Nov", "2019-Dec"]
RAW_DIR = "/Volumes/workspace/ecom/raw"

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.ecom;
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.ecom.raw;

# COMMAND ----------

# Download any month not already in the volume
import os
import urllib.request

for m in MONTHS:
    path = f"{RAW_DIR}/{m}.csv.gz"
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://data.rees46.com/datasets/marketplace/{m}.csv.gz", path)
        print("downloaded", m)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType

schema = StructType([
    StructField("event_time",    StringType(),  True),  # ends in " UTC"; parse to timestamp in the next step
    StructField("event_type",    StringType(),  True),  # view / cart / purchase
    StructField("product_id",    IntegerType(), True),  # max ~100M, fits in int
    StructField("category_id",   LongType(),    True),  # 19-digit IDs overflow int (max ~2.1e9)
    StructField("category_code", StringType(),  True),  # often null
    StructField("brand",         StringType(),  True),  # often null
    StructField("price",         DoubleType(),  True),  # fine for modelling; DecimalType is stricter for money
    StructField("user_id",       LongType(),    True),  # ~500M now, long is safe as IDs grow
    StructField("user_session",  StringType(),  True),  # UUID text
])

# One gz file = one task (gzip can't be split), so the months are read in parallel with each other
df = spark.read.csv([f"{RAW_DIR}/{m}.csv.gz" for m in MONTHS], header=True, schema=schema)

# COMMAND ----------

bronze = (
    df
    .withColumn("event_time", F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss 'UTC'"))
    .withColumn("event_date", F.to_date("event_time"))
    .withColumn("source_file", F.col("_metadata.file_name"))
)

bronze.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.ecom.bronze_events")

# Checks: no unparsed timestamps; rows per source file
b = spark.table("workspace.ecom.bronze_events")
assert b.filter(F.col("event_time").isNull()).count() == 0, "unparsed event_time values"
display(b.groupBy("source_file").count())

# COMMAND ----------

b = spark.table("workspace.ecom.bronze_events")

n0 = b.count()
s1 = b.filter(F.col("user_session").isNotNull());           n1 = s1.count()
s2 = s1.withColumn("is_valid_price", F.col("price") > 0)    # flag, keep the row
silver = s2.dropDuplicates();                               n3 = silver.count()
n_flagged = silver.filter(~F.col("is_valid_price")).count()

dq = spark.createDataFrame(
    [("bronze rows",                         n0, 0),
     ("drop missing user_session",           n1, n0 - n1),
     ("drop exact duplicates",               n3, n1 - n3),
     ("flag price <= 0 (kept, not removed)", n3, 0)],
    ["step", "rows_after", "rows_removed"],
)
display(dq)
print("rows flagged is_valid_price = false:", n_flagged)

silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.ecom.silver_events")
dq.write.format("delta").mode("overwrite").saveAsTable("workspace.ecom.silver_dq_summary")
