# Databricks notebook source
# Gold: one row per user per weekly cutoff. Features from events before the cutoff,
# label = at least one purchase in the 7 days after it.
from pyspark.sql import functions as F
from datetime import datetime, timedelta

def build_gold(cutoff_str, obs_days=30, horizon_days=7):
    cutoff      = datetime.strptime(cutoff_str, "%Y-%m-%d")
    obs_start   = cutoff - timedelta(days=obs_days)
    horizon_end = cutoff + timedelta(days=horizon_days)

    ev = spark.table("workspace.ecom.silver_events_sessionised")

    # 1. Split time FIRST: features only see the past, the label only sees the future
    past   = ev.filter((F.col("event_time") >= F.lit(obs_start)) & (F.col("event_time") < F.lit(cutoff)))
    future = ev.filter((F.col("event_time") >= F.lit(cutoff))    & (F.col("event_time") < F.lit(horizon_end)))
    assert past.agg(F.max("event_time")).first()[0] < cutoff, "LEAK: feature input has events at/after cutoff"

    is_view  = (F.col("event_type") == "view").cast("int")
    is_cart  = (F.col("event_type") == "cart").cast("int")
    is_purch = (F.col("event_type") == "purchase").cast("int")
    days_ago = (F.lit(cutoff).cast("long") - F.col("event_time").cast("long")) / 86400
    in_7d    = (days_ago <= 7).cast("int")
    in_1d    = (days_ago <= 1).cast("int")

    # 2. User features (past only)
    user = past.groupBy("user_id").agg(
        F.min(days_ago).alias("days_since_last_event"),
        F.min(F.when(is_cart == 1, days_ago)).alias("days_since_last_cart"),
        F.min(F.when(is_purch == 1, days_ago)).alias("days_since_last_purchase"),
        F.count("*").alias("n_events_30d"),
        F.sum(in_7d).alias("n_events_7d"),
        F.sum(in_1d).alias("n_events_1d"),
        F.countDistinct("event_date").alias("active_days_30d"),
        F.countDistinct("session_id").alias("n_sessions_30d"),
        F.sum(is_view).alias("n_views_30d"),
        F.sum(is_cart).alias("n_carts_30d"),
        F.sum(is_cart * in_7d).alias("n_carts_7d"),
        F.sum(is_purch).alias("n_purchases_30d"),
        F.sum(is_purch * in_7d).alias("n_purchases_7d"),
        F.sum(F.when(is_purch == 1, F.col("price"))).alias("spend_30d"),
        F.avg(F.when((is_view == 1) & F.col("is_valid_price"), F.col("price"))).alias("avg_price_viewed"),
        F.countDistinct("category_id").alias("n_categories_30d"),
        F.countDistinct("product_id").alias("n_products_30d"),
        F.countDistinct("brand").alias("n_brands_30d"),
    )
    user = (user
        .withColumn("carts_per_view",     F.col("n_carts_30d") / F.greatest(F.col("n_views_30d"), F.lit(1)))
        .withColumn("events_per_session", F.col("n_events_30d") / F.col("n_sessions_30d"))
        .withColumn("carted_not_bought_7d", ((F.col("n_carts_7d") > 0) & (F.col("n_purchases_7d") == 0)).cast("int"))
    )

    # 3. Visit features, re-summarised from PAST events only (visits are cut at the cutoff)
    visits = (past.groupBy("user_id", "session_id").agg(
                  ((F.max("event_time").cast("long") - F.min("event_time").cast("long")) / 60).alias("dur_min"),
                  F.max(is_cart).alias("has_cart"),
                  F.max(in_7d).alias("recent"))
              .groupBy("user_id").agg(
                  F.max("dur_min").alias("longest_visit_min"),
                  F.sum(F.col("has_cart") * F.col("recent")).alias("n_cart_visits_7d")))

    # 4. Label (future only): did they buy at least once in the next 7 days?
    label = (future.filter(F.col("event_type") == "purchase")
                   .select("user_id").distinct().withColumn("label", F.lit(1)))

    return (user.join(visits, "user_id", "left")
                .join(label,  "user_id", "left")
                .fillna({"label": 0})
                .withColumn("cutoff_date", F.lit(cutoff_str).cast("date")))

# Mondays with a full 30-day history and a complete 7-day label window inside Oct-Dec 2019.
# 4-18 Nov: training, validation and test weeks. 25 Nov-23 Dec: later weeks for monitoring:
# 25 Nov's label week contains Black Friday (29 Nov), 2 Dec's contains Cyber Monday;
# 23 Dec's label week ends on 30 Dec, the last full week in the December file.
CUTOFFS = ["2019-11-04", "2019-11-11", "2019-11-18",
           "2019-11-25", "2019-12-02", "2019-12-09", "2019-12-16", "2019-12-23"]
TABLE   = "workspace.ecom.gold_user_features"

for cutoff in CUTOFFS:
    gold = build_gold(cutoff)
    # Idempotent write: replace only this cutoff's rows, so a rerun never duplicates them
    if spark.catalog.tableExists(TABLE):
        gold.write.format("delta").mode("overwrite") \
            .option("replaceWhere", f"cutoff_date = '{cutoff}'").saveAsTable(TABLE)
    else:
        gold.write.format("delta").mode("overwrite").saveAsTable(TABLE)
    print("built", cutoff)

# Remove cutoffs from earlier development runs (e.g. 2019-10-21, which had only 20 days of history)
spark.sql(f"DELETE FROM {TABLE} WHERE cutoff_date NOT IN ({', '.join(repr(c) for c in CUTOFFS)})")

# COMMAND ----------

g = spark.table(TABLE)

# 1. One row per user per cutoff
dupes = g.groupBy("cutoff_date", "user_id").count().filter("count > 1").count()
assert dupes == 0, f"{dupes} duplicated user-cutoff rows"

# 2. Size and label rate per cutoff
display(g.groupBy("cutoff_date").agg(F.count("*").alias("users"), F.sum("label").alias("buyers"),
                                     F.avg("label").alias("label_rate")).orderBy("cutoff_date"))

# 3. Current business rule (B1) per cutoff
display(g.groupBy("cutoff_date", "carted_not_bought_7d")
         .agg(F.count("*").alias("users"), F.avg("label").alias("buy_rate"))
         .orderBy("cutoff_date", "carted_not_bought_7d"))
