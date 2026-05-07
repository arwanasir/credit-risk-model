# Credit Risk Model

A machine learning project for estimating customer credit risk and exposing predictions through a lightweight API layer. The repository combines feature engineering, model training, testing, and FastAPI-based inference to support a practical credit-scoring workflow.

## Overview

This project explores how customer-level data can be transformed into a usable credit-risk scoring pipeline. It is designed to cover the main stages of a development-oriented ML workflow:

- prepare and transform input features
- train and evaluate risk models
- validate data-processing behavior with tests
- expose predictions through a FastAPI service

The repository also reflects the business context of credit scoring, where model transparency, validation, and reproducibility matter alongside predictive performance.

## Repository Structure

```text
credit-risk-model/
├── notebooks/                # Exploratory and modeling notebooks
├── src/
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   └── api/
│       ├── main.py           # FastAPI inference service
│       └── pydentic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Core Components

- `src/feature_engineering.py`: transforms raw customer data into model-ready features
- `src/model.py`: model-related logic and reusable training helpers
- `src/train.py`: training workflow for generating a deployable model artifact
- `src/api/main.py`: prediction API for serving risk scores and related outputs
- `tests/test_data_processing.py`: tests for validating parts of the preprocessing workflow

## Tech Stack

- Python
- Pandas and NumPy
- scikit-learn
- pytest
- FastAPI
- Uvicorn
- MLflow
- Pydantic

## What This Project Demonstrates

- feature engineering for tabular risk modeling
- model training and evaluation in a structured repository
- separating training logic from serving logic
- wrapping an ML model in an API suitable for downstream integration
- awareness of documentation and interpretability requirements in regulated domains

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training or preprocessing workflows

Use the scripts in `src/` and the notebooks directory to reproduce the feature-engineering and training process.

### 3. Start the prediction API

```bash
uvicorn src.api.main:app --reload
```

Once running, the API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

## API Purpose

The API layer is intended to make trained-model outputs easier to consume in downstream applications. At a high level, it provides:

- a health or landing endpoint
- a prediction endpoint that accepts feature values
- a structured response containing risk-oriented output values

## Notes

This repository blends model development and deployment concerns in one place. The strongest signal for reviewers is that it is not only a notebook project: it also includes reusable Python modules, tests, and an API layer for inference.
