#!/usr/bin/env python3
import argparse, pandas as pd, pickle

def build_parser():
    p = argparse.ArgumentParser(description="Run inference on feature-only CSV.")
    p.add_argument("--features", required=True)    # features without y
    p.add_argument("--model-in", default="models/model.pkl")
    p.add_argument("--output", required=True)      # predictions csv
    p.add_argument("--pred-col", default="prediction")
    return p

def main():
    a = build_parser().parse_args()
    X = pd.read_csv(a.features)
    with open(a.model_in, "rb") as f:
        model = pickle.load(f)
    preds = model.predict(X.values)
    out = pd.DataFrame({a.pred_col: preds})
    out.to_csv(a.output, index=False)

if __name__ == "__main__":
    main()

