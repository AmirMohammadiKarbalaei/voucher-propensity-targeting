# Databricks notebook source
# Sessions: a new visit starts after more than 30 minutes of inactivity (window functions).
from pyspark.sql import functions as F
from pyspark.sql.window import Window

s = spark.table("workspace.ecom.silver_events")

w = Window.partitionBy("user_id").orderBy("event_time")       # each user's events in time order
w_running = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)

events = (
    s
    .withColumn("prev_time", F.lag("event_time").over(w))                       # step 1
    .withColumn("gap_min", (F.col("event_time").cast("long") - F.col("prev_time").cast("long")) / 60)
    .withColumn("is_new_session",                                               # step 2
                F.when(F.col("prev_time").isNull() | (F.col("gap_min") > 30), 1).otherwise(0))
    .withColumn("session_num", F.sum("is_new_session").over(w_running))         # step 3
    .withColumn("session_id", F.concat_ws("_", "user_id", "session_num"))
)

events.explain()   # the plan shows one shuffle: Exchange hashpartitioning(user_id)

events.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
      .saveAsTable("workspace.ecom.silver_events_sessionised")


# COMMAND ----------


# One row per visit: the table later features are built from
sessions = (
    spark.table("workspace.ecom.silver_events_sessionised")
    .groupBy("user_id", "session_id")
    .agg(
        F.min("event_time").alias("session_start"),
        F.max("event_time").alias("session_end"),
        F.count("*").alias("n_events"),
        F.sum((F.col("event_type") == "view").cast("int")).alias("n_views"),
        F.sum((F.col("event_type") == "cart").cast("int")).alias("n_carts"),
        F.sum((F.col("event_type") == "purchase").cast("int")).alias("n_purchases"),
    )
    .withColumn("duration_min",
                (F.col("session_end").cast("long") - F.col("session_start").cast("long")) / 60)
)
sessions.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable("workspace.ecom.silver_sessions")



# COMMAND ----------

from pyspark.sql import functions as F

ev = spark.table("workspace.ecom.silver_events_sessionised")
ss = spark.table("workspace.ecom.silver_sessions")

# 1. Our sessions vs the website's
print("our sessions:    ", ss.count())
print("site sessions:   ", ev.select("user_session").distinct().count())
print("users:           ", ev.select("user_id").distinct().count())

# 2. What does a typical visit look like?
display(ss.select(
    F.expr("percentile(n_events, array(0.5, 0.9, 0.99))").alias("events_p50_p90_p99"),
    F.expr("percentile(duration_min, array(0.5, 0.9, 0.99))").alias("minutes_p50_p90_p99"),
))

# 3. Rule check: within a session, no gap should exceed 30 minutes (expect 0)
print("broken gaps:", ev.filter((F.col("is_new_session") == 0) & (F.col("gap_min") > 30)).count())
