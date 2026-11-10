"""
Weekly propensity pipeline.

1. run_databricks_job: Databricks job `ecom_pipeline` rebuilds bronze -> silver -> sessions -> gold (Spark).
2. score_users: score every user for this week (placeholder, see below).
3. day0_checks: the monitoring rule's Monday checks (data health, input drift, output drift).
   Exit code 1 = STOP (the data is broken), so send_vouchers never starts. ALERT = send anyway
   and a person follows the runbook (docs/monitoring/runbook.md).
4. send_vouchers: top 50,000 users get a voucher (placeholder).
5. day7_checks: last week's answers are now in: calibration and ranking quality. Runs whether or
   not this week's send went ahead.

Placeholders: scoring and sending are EmptyOperators: in this project the scores are written by
`monitoring/05_weekly_checks.py replay` (data/scores/<week>.parquet); in production they'd be a batch
scoring job and a call to the voucher system.

The data is from November 2019, so the week checked is a parameter, not the run date.
"""
from datetime import datetime, timedelta

from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG, Param

# Separate virtual environment, so the checks' dependencies can't clash with Airflow's
CHECKS = ("PYTHONDONTWRITEBYTECODE=1 /home/airflow/monitor-venv/bin/python "
          "/opt/airflow/monitoring/05_weekly_checks.py")

default_args = {
    "owner": "amir",
    "retries": 2,                          # try again twice before failing
    "retry_delay": timedelta(minutes=5),   # wait between attempts (e.g. a brief network blip)
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    dag_id="ecom_pipeline",
    description="Spark tables on Databricks, then score, check and send",
    doc_md=__doc__,
    schedule="0 6 * * 1",                  # Mondays 06:00 UTC, before the weekly voucher send
    start_date=datetime(2026, 10, 1),
    catchup=False,                         # don't run for every missed Monday since start_date
    max_active_runs=1,                     # never two runs at once
    default_args=default_args,
    params={
        "week": Param("2019-11-18", type="string", description="cut-off being scored and sent this run"),
        "last_week": Param("2019-11-11", type="string", description="cut-off whose answers are now in"),
    },
    tags=["ecom", "weekly"],
) as dag:

    run_databricks_job = DatabricksRunNowOperator(
        task_id="run_databricks_job",
        databricks_conn_id="databricks_default",
        job_name="ecom_pipeline",
    )

    score_users = EmptyOperator(task_id="score_users")

    day0_checks = BashOperator(
        task_id="day0_checks",
        bash_command=f"{CHECKS} day0 --week {{{{ params.week }}}}",
        retries=0,                         # a STOP is a finding, not a blip: don't retry it
    )

    send_vouchers = EmptyOperator(task_id="send_vouchers")

    day7_checks = BashOperator(
        task_id="day7_checks",
        bash_command=f"{CHECKS} day7 --week {{{{ params.last_week }}}}",
        trigger_rule="all_done",           # still check last week even if today's run stopped
    )

    run_databricks_job >> score_users >> day0_checks >> send_vouchers
    day0_checks >> day7_checks
