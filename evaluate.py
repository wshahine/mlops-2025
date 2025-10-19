#!/usr/bin/env python3
import argparse, pandas as pd, json, pickle
from sklearn.metrics import accuracy_score

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate a saved model.")
    p.add_argument("--features", required=True)   # features CSV with y
    p.add_argument("--target", default="Survived")
    p.add_argument("--model-in", default="models/model.pkl")
    p.add_argument("--metrics-out", default="metrics.json")
    return p

def main():
    a = build_parser().parse_args()
    df = pd.read_csv(a.features)
    y = df[a.target].values
    X = df.drop(columns=[a.target]).values
    with open(a.model_in, "rb") as f:
        model = pickle.load(f)
    acc = accuracy_score(y, model.predict(X))
    metrics = {"accuracy": float(acc)}
    with open(a.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == "__main__":
    main()
