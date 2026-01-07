# VT2 - Uncorrelated Encounter Model for Swiss Free Route Airspace

This repository contains the full implementation for the specialization project **"Uncorrelated Encounter Model for Swiss Free Route Airspace"**, developed at the **Zurich University of Applied Sciences (ZHAW)**, Center for Aviation (ZAV). The project explores **short-term aircraft trajectory prediction in en-route airspace** using a continuous **Dynamic Bayesian Network** model trained on historical ADS-B data. By focusing on scenarios where air traffic control intervention is minimal and only the dynamics of the aircraft determine the evolution of encounters, the model provides a foundation for analyzing safety-critical situations where avoidance systems such as TCAS are already challenged.

---

## 🚀 Project Pipeline

The project is structured as a three-step pipeline:

1.  **Dataset Creation**: Run `01_dataset_creation.ipynb` to process raw ADS-B parquet files and generate normalized sets.
2.  **Model Training**: Use `train_bn.py` to train the Residual Gaussian Bayesian Network.
3.  **Evaluation**: Use `02_model_evaluation.ipynb` to analyze results (metrics, PIT histograms, visualizations).

---

## 📁 Project Structure

```
VT_2/
├── 01_dataset_creation.ipynb   # Data engineering and sampling
├── train_bn.py                 # Main training script (Gaussian BN)
├── 02_model_evaluation.ipynb    # Metrics and visualization
├── utils/                      # Core project package
│   ├── utils.py                # Geometry and dataset logic
│   ├── training_utils.py       # Model architecture and training loops
│   └── metrics.py              # Statistical evaluation tools
├── traffic.yml                 # Conda environment configuration
└── README.md
```

---

## 📦 Environment

* `traffic.yml` — using the [`traffic`](https://traffic-viz.github.io/) library.

To recreate them:

```bash
conda env create -f traffic.yml
```

---

## 📚 Citation

> **Alex Fustagueras**. <br>
> *Uncorrelated Encounter Model for Swiss Free Route Airspace*. <br>
> Specialization Project, ZHAW Centre for Aviation, January 2026