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

