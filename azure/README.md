# Azure ML: train and deploy the propensity model

The winning LightGBM model (settings tuned locally in `modelling/02_boosting.py`) is retrained as an
Azure ML job and served from a managed online endpoint. Everything below is done in the Azure portal
and Azure ML studio; the only local step is preparing the files.

**Cost controls:** Free Trial subscription (spending limit on), a 10/month budget alert created before
any resource, one resource group (`rg-ecom-ml`) so everything can be deleted at once, a training
cluster that scales to 0 machines, and the endpoint deleted at the end of each session.

## Files
| File | What it is |
|---|---|
| `prepare_data.py` | Local: writes `data/azure/ecom_gold/{train,test}.parquet` and `sample_request.json` |
| `src/train.py` | The training job: fit LightGBM, log test metrics, save an MLflow model to `outputs/model` |
| `src/params.json` | The tuned LightGBM settings (copied from `reports/best_params.json`) |
| `conda.yaml` | The job's Python libraries (the Azure ML "environment") |
| `sample_request.json` | Three test-week users in the format the endpoint expects |

## Steps

### 0. Prepare the files (local)
`uv run python azure/prepare_data.py`

### 1. Workspace and budget
- Portal (portal.azure.com) → **Resource groups** → `rg-ecom-ml`: the workspace and its helpers
  (storage account, key vault, Application Insights, Log Analytics).
- Portal → **Cost Management** → **Budgets**: `ecom-budget-10`. **Cost analysis**: spend so far.
- Open the workspace `ecom-ml` → **Launch studio** (ml.azure.com). The rest happens in the studio.

### 2. Data asset
Studio → **Data** → **+ Create**
- Name `ecom_gold`, type **Folder (uri_folder)**
- From **local files** → datastore `workspaceblobstore` → upload the folder `data/azure/ecom_gold`

### 3. Compute cluster
Studio → **Compute** → **Compute clusters** → **+ New**
- Dedicated, CPU, size **Standard_DS2_v2** (2 cores, 7 GB) or any 2-core size listed
- Name `cpu-cluster`, **minimum nodes 0**, maximum nodes 1, idle seconds before scale down **120**

### 4. Environment
Studio → **Environments** → **Custom environments** → **+ Create**
- Name `lgbm-train`, source **Use existing docker image with optional conda file**
- Image `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest`, upload `azure/conda.yaml`

### 5. Training job
Studio → **Jobs** → **+ Create job** → **Custom**
- Code: upload the folder `azure/src`
- Compute `cpu-cluster`, environment `lgbm-train`
- Input: name `data`, type data asset, `ecom_gold`
- Command: `python train.py --data ${{inputs.data}}`

The first run builds the environment image (10–20 min, once). Check **Metrics**: test precision in the
top 50,000 should be 0.2476, the same as the local run.

### 6. Register the model
Job → **Outputs + logs** → `outputs/model` → **Register model**, type **MLflow**, name `propensity_model`.

**Note:** Azure ML's no-code MLflow deployment needs `azureml-ai-monitoring` and
`azureml-inference-server-http` in the model's own conda env. Without them the scoring container
crashes on start-up (`ModuleNotFoundError: No module named 'azureml'`) and the deployment fails after
about 30 minutes. `src/train.py` now adds both; `propensity_model` v2 is the v1 model file with them added.

### 7. Deploy (billed per hour from here)
Either in the studio: Models → `propensity_model` → **Deploy** → **Real-time endpoint**, or with the CLI:
```
az ml online-endpoint create -n ecom-propensity-test --auth-mode key -g rg-ecom-ml -w ecom-ml
az ml online-deployment create -f azure/deployment-blue.yml -g rg-ecom-ml -w ecom-ml --all-traffic
```
- Size **Standard_D2as_v4** (2 cores, 8 GB, about $0.11/h), **1 instance** (the wizard defaults to 3).
  The free trial has 4 cores; one 2-core instance reserves 4 (Azure keeps 20% spare, rounded up to a
  whole machine), so the cluster must be at 0 nodes.
- No scoring script needed: MLflow models are served directly. First deployment takes 15–30 minutes
  (image build). If it hangs, read the logs: `az ml online-deployment get-logs -n blue -e <endpoint> ...`

### 8. Test
Endpoint → **Test** tab → paste `sample_request.json`. The answer is one pair per user:
`[chance of not buying, chance of buying]`; expected about `0.063, 0.239, 0.406`.

Streamlit demo: put the endpoint's scoring URL and primary key in a git-ignored `.env`
(`AZURE_ML_ENDPOINT_URL=...`, `AZURE_ML_ENDPOINT_KEY=...`), then `uv run streamlit run app/streamlit_app.py`.
Screenshots: `docs/streamlit/`.

### 9. Delete the endpoint
Endpoints → select → **Delete** (or `az ml online-endpoint delete -n ecom-propensity-test --yes ...`).
Check Compute shows 0 nodes. To remove everything: `az group delete -n rg-ecom-ml --yes`.
