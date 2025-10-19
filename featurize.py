#!/usr/bin/env python3
import argparse, pandas as pd
from sklearn.preprocessing import OneHotEncoder

def build_parser():
    p = argparse.ArgumentParser(description="Feature engineering.")
    p.add_argument("--input", required=True)   # preprocessed CSV (may include y)
    p.add_argument("--output", required=True)  # features CSV (X plus y)
    p.add_argument("--target", default="Survived")
    return p

def main():
    a = build_parser().parse_args()
    df = pd.read_csv(a.input)

    y = df[a.target] if a.target in df.columns else None

    # Example features (match your notebook exactly where possible)
    num_cols = [c for c in df.select_dtypes("number").columns if c != a.target]
    cat_cols = [c for c in df.select_dtypes("object").columns]

    X_num = df[num_cols].copy()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = pd.DataFrame(ohe.fit_transform(df[cat_cols]))
    X_cat.columns = ohe.get_feature_names_out(cat_cols)

    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    if y is not None:
        X[a.target] = y.values
    X.to_csv(a.output, index=False)

if __name__ == "__main__":
    main()
