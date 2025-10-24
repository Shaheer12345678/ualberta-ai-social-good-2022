#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

def synthesize(n=2000):
    # Features: temp, humidity, wind, vegetation_dryness
    temp = np.random.normal(25, 7, n)
    humidity = np.random.uniform(10, 80, n)
    wind = np.random.normal(15, 6, n)
    dryness = np.random.uniform(0, 1, n)
    # Risk label (synthetic rule + noise)
    logit = 0.08*temp - 0.04*humidity + 0.09*wind + 2.0*dryness - 4.0
    prob = 1/(1+np.exp(-logit))
    y = (prob > 0.5).astype(int)
    X = pd.DataFrame({"temp":temp,"humidity":humidity,"wind":wind,"dryness":dryness})
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    X, y = synthesize()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    if args.train:
        clf.fit(Xtr, ytr)
    if args.evaluate:
        if len(clf.feature_importances_)==0:
            clf.fit(Xtr, ytr)
        pred = clf.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, pred)
        print("AUC:", round(auc, 4))
        print(classification_report(yte, (pred>0.5).astype(int)))
if __name__ == "__main__":
    main()
