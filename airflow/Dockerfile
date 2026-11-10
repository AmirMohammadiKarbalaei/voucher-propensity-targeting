FROM apache/airflow:3.3.2

# LightGBM needs the OpenMP runtime, which the slim base image doesn't include
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
USER airflow

# Airflow providers go into Airflow's own environment, pinned to the same Airflow version
COPY requirements.txt .
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r requirements.txt

# Monitoring checks (monitoring/05_weekly_checks.py): same library versions as the project's uv.lock
RUN python -m venv /home/airflow/monitor-venv \
 && /home/airflow/monitor-venv/bin/pip install --no-cache-dir \
      "lightgbm==4.7.0" "pandas==3.0.6" "numpy==2.4.6" "scikit-learn==1.9.1" "pyarrow==25.0.1"
