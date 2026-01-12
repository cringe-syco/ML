Below is a **complete MLflow cheatsheet in Markdown**, ready to drop into your repo (`MLFLOW_CHEATSHEET.md`).
It covers **tracking, autologging, models, registry, serving, projects, and production tips**, with runnable examples.

---

# 🧪 MLflow Cheatsheet

## Table of Contents

* [Installation](#installation)
* [MLflow Components](#mlflow-components)
* [Tracking Experiments](#tracking-experiments)
* [Autologging](#autologging)
* [Logging Artifacts](#logging-artifacts)
* [Model Logging & Loading](#model-logging--loading)
* [Model Registry](#model-registry)
* [Serving Models](#serving-models)
* [MLflow Projects](#mlflow-projects)
* [MLflow with Popular Frameworks](#mlflow-with-popular-frameworks)
* [Remote Tracking Server](#remote-tracking-server)
* [CLI Commands](#cli-commands)
* [Best Practices](#best-practices)
* [Common Pitfalls](#common-pitfalls)

---

## Installation

```bash
pip install mlflow
```

Start UI:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

---

## MLflow Components

| Component      | Purpose                        |
| -------------- | ------------------------------ |
| Tracking       | Log params, metrics, artifacts |
| Projects       | Reproducible ML runs           |
| Models         | Standardized model packaging   |
| Model Registry | Model lifecycle management     |

---

## Tracking Experiments

### Basic Tracking

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.92)
```

### Nested Runs

```python
with mlflow.start_run(run_name="parent"):
    with mlflow.start_run(run_name="child", nested=True):
        mlflow.log_metric("loss", 0.2)
```

### Tags

```python
mlflow.set_tag("team", "ml")
mlflow.set_tag("env", "dev")
```

---

## Autologging

```python
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

with mlflow.start_run():
    model.fit(X_train, y_train)
```

Supported frameworks:

* sklearn
* TensorFlow / Keras
* PyTorch
* XGBoost
* LightGBM
* Statsmodels

---

## Logging Artifacts

### Single File

```python
mlflow.log_artifact("model.pkl")
```

### Directory

```python
mlflow.log_artifacts("outputs/")
```

### Temporary Artifact

```python
with open("info.txt", "w") as f:
    f.write("Run details")

mlflow.log_artifact("info.txt")
```

---

## Model Logging & Loading

### Log a Model (Sklearn)

```python
import mlflow.sklearn

mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="MyModel"
)
```

### Load by Run ID

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "runs:/<run_id>/model"
)
preds = model.predict(X)
```

### Load from Registry

```python
model = mlflow.pyfunc.load_model(
    "models:/MyModel/Production"
)
```

---

## Model Registry

### Register Model Manually

```python
mlflow.register_model(
    "runs:/<run_id>/model",
    "MyModel"
)
```

### Transition Stage

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)
```

Stages:

* None
* Staging
* Production
* Archived

---

## Serving Models

### Local REST API

```bash
mlflow models serve \
  -m models:/MyModel/Production \
  -p 5001
```

### Request

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[1.2, 3.4, 5.6, 7.8]]}'
```

---

## MLflow Projects

### `MLproject`

```yaml
name: my_project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.1}
    command: "python train.py --alpha {alpha}"
```

### Run Project

```bash
mlflow run . -P alpha=0.5
```

---

## MLflow with Popular Frameworks

### PyTorch

```python
import mlflow.pytorch

with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
```

### TensorFlow / Keras

```python
import mlflow.tensorflow

mlflow.tensorflow.autolog()
model.fit(X, y)
```

### XGBoost

```python
import mlflow.xgboost

mlflow.xgboost.autolog()
```

---

## Remote Tracking Server

### Start Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 \
  --port 5000
```

### Connect Client

```python
mlflow.set_tracking_uri("http://server:5000")
```

---

## CLI Commands

```bash
mlflow experiments list
mlflow runs list
mlflow models list
mlflow models describe -m MyModel
mlflow models delete -m MyModel -v 3
```

---

## Best Practices

✅ Use **autologging**
✅ Name experiments clearly
✅ Log data versions
✅ Register only validated models
✅ Separate dev / staging / prod tracking servers
✅ Store artifacts in S3 / GCS / Azure Blob

---

## Common Pitfalls

❌ Forgetting `mlflow.start_run()`
❌ Logging large raw datasets
❌ Overwriting model names
❌ No experiment naming
❌ Using local file store in production

---

## Typical MLflow Workflow

```text
Train → Log → Compare → Register → Stage → Serve
```

---

## References

* [https://mlflow.org](https://mlflow.org)
* [https://github.com/mlflow/mlflow](https://github.com/mlflow/mlflow)

---
