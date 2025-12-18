# S&P 500 News and Market Sentiment Analytics

This repository contains the code for an exam project in **News and Market Sentiment Analytics**.

## Objective
To evaluate whether financial news headlines contain predictive information about future S&P 500 movements, and to compare fast sentiment models with modern transformer-based approaches under realistic time-series evaluation.

## Data
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024

Place the CSV file in:
data/sp500_news.csv

Expected columns:
- `Date` – trading date
- `Title` – news headline
- `CP` – S&P 500 closing price

OR run get_data.py in data/ folder with applicable changes.

## Reproducibility
- Time-series split (no random shuffling)
- Fixed preprocessing pipeline
- Deterministic models
- Python ≥ 3.9 recommended

## 
Main experiment is in src\experiment.py

## Environment setup (GPU)

```bash
conda env create -f environment.yml
conda activate sp500sent
python src/run_experiment.py


## Run
```bash
pip install -r requirements.txt
python src/run_experiment.py
