# 🚲 Bike Sharing Demand – MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=flat&logo=mlflow&logoColor=blue)
![Optuna](https://img.shields.io/badge/Optuna-white.svg?style=flat&logo=optuna&logoColor=blue)
![Hydra](https://img.shields.io/badge/Hydra-configuration-orange)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-purple)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

This repository contains a full MLOps pipeline for predicting bike sharing demand using a **RandomForestRegressor**. The project demonstrates the integration of modern tools for configuration management, hyperparameter tuning, experiment tracking, data versioning, and CI/CD.

## 🌟 Key Features

- **Configuration Management**: Powered by [Hydra](https://hydra.cc/), allowing clean and hierarchical configurations (`config/config.yaml`).
- **Hyperparameter Optimization (HPO)**: Handled by [Optuna](https://optuna.org/) using the TPE sampler to find the optimal `max_depth` and `n_estimators`.
- **Experiment Tracking**: Integrated with [MLflow](https://mlflow.org/) utilizing nested runs. The *parent run* tracks the Optuna study, while *child runs* capture individual trial metrics (e.g., RMSE) and parameters.
- **CI/CD & Quality Gates**: Automated testing via `pytest`. The pipeline checks data schemas before training and enforces a **Quality Gate (RMSE < 50.0)** before validating the final model.
- **Continuous Machine Learning (CML)**: Integrated with GitHub Actions and [Iterative CML](https://cml.dev/) to automatically generate Markdown reports of the model’s metrics directly on Commits/Pull Requests.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/bogdanluginin/mlops.git
cd mlops
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Hyperparameter Optimization

Execute the Hydra-decorated pipeline:

```bash
python src/optimize.py
```
*This will automatically start Optuna optimization, perform Cross-Validation, save the best `metrics.json`, and track the whole process in MLflow.*

### 3. Track Experiments in MLflow

Start the local MLflow tracking server:

```bash
mlflow ui
```
*Navigate to `http://127.0.0.1:5000` to view all your runs within the **Bike_Sharing_Optuna** experiment.*

### 4. Run Tests (Quality Gate)

To manually run the checks for the dataset schema and the trained model's performance:

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 🏗️ Project Structure

```text
├── .github/workflows/cml.yaml   # CI/CD instructions for GitHub Actions & CML reports
├── config/
│   └── config.yaml              # Hydra configuration parameters
├── data/                        # Dataset storage
│   ├── raw/
│   └── prepared/
├── src/
│   └── optimize.py              # Main script (Model training + Optuna + MLflow)
├── tests/
│   └── test_pipeline.py         # Pytest definitions for Data Schema and Quality Gates
├── metrics.json                 # Auto-generated metrics used by Quality Gate
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
