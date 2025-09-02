# Iris Classifier (Decision Tree)

## Overview
End‑to‑end ML example from Digital Marketing Mastery Module → builds a decision‑tree classifier on the classic Iris dataset using scikit‑learn.

## Quick start
```bash
git clone https://github.com/<YOUR_USERNAME>/iris-classifier.git
cd iris-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py

## Project structure

iris-classifier/
├── data/ # empty (Iris dataset loaded via scikit-learn)
├── notebooks/ # Jupyter notebooks (e.g., iris_model.ipynb)
├── outputs/ # Generated model artifacts and plots
├── src/ # Python scripts
│ └── train.py # Main training script
├── tests/ # Unit tests
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt